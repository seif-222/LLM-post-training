"""Pinned question manifests and resumable evaluation; no network/compute on import."""
import json
import math
from collections import Counter
from pathlib import Path
from rl.campaign import digest, evaluation_requests, isolated_rng
from rl.checkpointing import atomic_json, file_hash, commit_retry


def validate_manifest(manifest):
    questions = manifest['questions']
    ids = [q['id'] for q in questions]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('missing or duplicate manifest IDs')
    if any(not math.isfinite(q['answer']) for q in questions):
        raise ValueError('nonfinite gold')
    if manifest['dataset'] == 'openai/gsm8k/main/test' and set(ids) != set(range(1319)):
        raise ValueError('GSM8K manifest requires exact 1319 IDs')
    if manifest['dataset'] == 'apple/GSM-Symbolic/main/test':
        counts = Counter(q['template_id'] for q in questions)
        if len(questions) != 5000 or set(counts) != set(range(100)) or set(counts.values()) != {50}:
            raise ValueError('Symbolic requires 100 templates x 50 instances')
        if {(q['template_id'], q['instance_id']) for q in questions} != {(t, i) for t in range(100) for i in range(50)}:
            raise ValueError('Symbolic template/instance mismatch')
    return manifest


def symbolic_manifest(rows, gsm8k, revision, checksum):
    from rl.data import extract_final_answer
    questions = []
    for row in rows:
        original_id = row['original_id']
        if not 0 <= original_id < len(gsm8k) or row['original_question'] != gsm8k[original_id]['question']:
            raise ValueError(f'Symbolic original question mismatch: {original_id}')
        questions.append({'id': f'{row["id"]}:{row["instance"]}', 'template_id': row['id'],
                          'instance_id': row['instance'], 'original_id': original_id,
                          'question': row['question'], 'answer': extract_final_answer(row['answer'])})
    return validate_manifest({'dataset': 'apple/GSM-Symbolic/main/test', 'revision': revision,
                              'source_sha256': checksum, 'questions': questions})


def validate_records(records, questions, k):
    expected = {q['id']: q for q in questions}
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('missing, duplicate or extra evaluation IDs')
    for record in records:
        q = expected[record['id']]
        if record['question'] != q['question'] or record['answer'] != q['answer']:
            raise ValueError('evaluation question/gold mismatch')
        samples = record['samples']
        if len(samples) != k:
            raise ValueError('evaluation sample count mismatch')
        for sample in samples:
            if sample['length'] != len(sample['token_ids']) or sample['finish_reason'] not in ('stop', 'length'):
                raise ValueError('invalid token or finish record')
        if record['n_correct'] != sum(s['correct_strict'] for s in samples):
            raise ValueError('evaluation success count mismatch')
    return records


def evaluate_manifest(engine, manifest, output, k, sampled=True, model_identity=None, commit=None, guard=None):
    from rl.reward import score
    validate_manifest(manifest)
    if sampled and k not in (32, 64) and manifest['dataset'] == 'openai/gsm8k/main/test':
        raise ValueError('main evaluation requires uniform k=32 or k=64')
    if not sampled and k != 1:
        raise ValueError('greedy requires k=1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {'manifest_hash': digest(manifest), 'k': k, 'sampled': sampled, 'model': model_identity,
              'seed_derivation_version': 1, 'chunk_size': 32}
    if (output/'config.json').exists():
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('evaluation configuration mismatch')
    else:
        atomic_json(output/'config.json', config)
    questions, all_records = manifest['questions'], []
    expected_chunks = {f'{i//32:04d}.json' for i in range(0, len(questions), 32)}
    if {p.name for p in (output/'chunks').glob('*.json')} - expected_chunks:
        raise ValueError('extra evaluation chunk')
    for start in range(0, len(questions), 32):
        chunk = questions[start:start+32]
        path = output/'chunks'/f'{start//32:04d}.json'
        marker = path.with_suffix('.sha256')
        if path.exists() and marker.exists():
            if marker.read_text() != file_hash(path):
                raise ValueError('evaluation chunk checksum mismatch')
            records = validate_records(json.loads(path.read_text()), chunk, k)
        else:
            if path.exists():
                import shutil, uuid
                quarantine = output/'incomplete-chunks'/(path.name+'.'+uuid.uuid4().hex)
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, quarantine)
            if guard:
                guard()
            requests = evaluation_requests(manifest['dataset'], [q['id'] for q in chunk])
            with isolated_rng():
                if sampled:
                    rollout = engine.sample(chunk, k, 1.0, requests=requests)
                    samples = []
                    P = rollout['prompt_len']
                    for i, text in enumerate(rollout['texts']):
                        length = int(rollout['lengths'][i])
                        truncated = bool(rollout['truncated'][i])
                        samples.append({'text': text, 'length': length, 'truncated': truncated,
                                        'token_ids': rollout['input_ids'][i, P:P+length].tolist(),
                                        'finish_reason': 'length' if truncated else 'stop'})
                    groups = [samples[i*k:(i+1)*k] for i in range(len(chunk))]
                else:
                    groups = [[g] for g in engine.greedy(chunk)]
            records = []
            for q, group, request in zip(chunk, groups, requests):
                scored = [{**g, **score(g['text'], q['answer']), 'request': request} for g in group]
                records.append({**q, 'samples': scored, 'n_correct': sum(s['correct_strict'] for s in scored)})
            validate_records(records, chunk, k)
            atomic_json(path, records)
            marker.write_text(file_hash(path))
            commit_retry(commit)
        all_records.extend(records)
    validate_records(all_records, questions, k)
    from rl.eval import passk_summary
    flat = [sample for record in all_records for sample in record['samples']]
    summary = {'n': len(all_records), 'k': k,
               'accuracy_strict': sum(s['correct_strict'] for s in flat)/len(flat),
               'accuracy_lenient': sum(s['correct_lenient'] for s in flat)/len(flat),
               'answer_source': {str(source):sum(s['source']==source for s in flat)/len(flat) for source in ('boxed','hash','last',None)},
               'mean_length': sum(s['length'] for s in flat)/len(flat),
               'cutoff_fraction': sum(s['truncated'] for s in flat)/len(flat),
               'passk': passk_summary([r['n_correct'] for r in all_records], k),
               'complete': True, 'config': config}
    atomic_json(output/'summary.json', summary)
    commit_retry(commit)
    return summary
