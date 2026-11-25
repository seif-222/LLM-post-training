"""Immutable, checksum-validated checkpoint generations and durable log chunks."""
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from rl.campaign import canonical, finite, rng_state, restore_rng


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('wb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def commit_retry(commit):
    if commit:
        try:
            commit()
        except Exception:
            commit()  # exactly one retry, no training progression


def save(path, model, opt, sched, stream, reward_gen, step, extra, commit=None):
    import torch
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    finite([p.detach() for p in model.parameters()], 'parameters')
    finite(opt.state_dict(), 'optimizer')
    generation = root / f'update-{step:04d}-{uuid.uuid4().hex}'
    generation.mkdir()
    model.save_pretrained(str(generation))
    state = {'opt': opt.state_dict(), 'sched': sched.state_dict(), 'stream': stream.state_dict(),
             'reward_gen': reward_gen.get_state(), 'step': step, 'rng': rng_state(), **extra}
    torch.save(state, generation / 'state.pt')
    hashes = {str(p.relative_to(generation)): file_hash(p) for p in generation.rglob('*') if p.is_file()}
    manifest = {'step': step, 'config_hash': extra.get('config_hash'), 'checksums': hashes}
    atomic_json(generation / 'manifest.json', manifest)
    validate(generation, extra.get('config_hash'), require_complete=False)
    commit_retry(commit)
    atomic_json(generation / 'COMPLETE.json', {'manifest_sha256': file_hash(generation / 'manifest.json')})
    commit_retry(commit)
    atomic_json(root / 'latest.json', {'generation': generation.name})
    commit_retry(commit)
    # Published generations are ordered by successful update; incomplete writes remain evidence.
    complete = sorted((p for p in root.glob('update-*') if (p / 'COMPLETE.json').exists()),
                      key=lambda p: (json.loads((p/'manifest.json').read_text())['step'], p.stat().st_mtime_ns))
    for old in complete[:-2]:
        shutil.rmtree(old)
    return generation


def validate(path, config_hash=None, require_complete=True):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if require_complete:
        complete = json.loads((path / 'COMPLETE.json').read_text())
        if complete['manifest_sha256'] != file_hash(path / 'manifest.json'):
            raise ValueError('checkpoint manifest checksum mismatch')
    if config_hash is not None and manifest['config_hash'] != config_hash:
        raise ValueError('checkpoint configuration mismatch')
    for name, expected in manifest['checksums'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(path / name) != expected:
            raise ValueError(f'checkpoint checksum mismatch: {name}')
    return manifest


def load(path, model, opt, sched, stream, reward_gen, config_hash=None):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    root = Path(path)
    pointer = json.loads((root / 'latest.json').read_text())  # missing never becomes a fresh run
    name = pointer['generation']
    if Path(name).name != name:
        raise ValueError('invalid checkpoint generation')
    generation = root / name
    validate(generation, config_hash)
    state = torch.load(generation / 'state.pt', weights_only=False, map_location='cpu')
    finite(state['opt'], 'optimizer')
    weights = load_file(generation / 'adapter_model.safetensors')
    finite(weights, 'adapter')
    set_peft_model_state_dict(model, weights)
    opt.load_state_dict(state['opt'])
    sched.load_state_dict(state['sched'])
    stream.load_state_dict(state['stream'])
    reward_gen.set_state(state['reward_gen'])
    restore_rng(state['rng'])
    return state


class ChunkLog:
    """Each flush publishes one closed gzip member; partial tails are never canonical."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.buffer, self.index = [], 0

    def write(self, text):
        self.buffer.append(text)

    def flush(self):
        if not self.buffer:
            return
        import gzip
        path = self.directory / f'{self.index:06d}.jsonl.gz'
        tmp = path.with_suffix('.tmp')
        with gzip.open(tmp, 'wt') as f:
            f.writelines(self.buffer)
        os.replace(tmp, path)
        atomic_json(path.with_suffix('.complete.json'), {'sha256': file_hash(path)})
        self.buffer.clear()
        self.index += 1

    def close(self):
        self.flush()


class MirrorLog:
    def __init__(self, legacy, chunks):
        self.legacy, self.chunks = legacy, chunks

    def write(self, text):
        self.chunks.write(text)
        self.legacy.write(text)

    def flush(self):
        self.chunks.flush()
        self.legacy.flush()

    def close(self):
        self.chunks.close()
        self.legacy.close()


def restore_logs(root, lineage):
    """Rebuild views ONLY from the chunks named by a validated checkpoint's lineage."""
    import gzip
    root = Path(root)
    for kind, filename, opener in (('metrics', 'metrics.jsonl', open), ('hard', 'hard.jsonl', open),
                                   ('samples', 'samples.jsonl.gz', gzip.open)):
        path = root/filename
        if path.exists():
            shutil.copy2(path, path.with_name(path.name+'.attempt-'+uuid.uuid4().hex))
        with opener(path, 'wt') as output:
            for attempt, counts in lineage.items():
                if Path(attempt).name != attempt:
                    raise ValueError('invalid attempt lineage')
                for index in range(counts.get(kind, 0)):
                    chunk = root/'attempts'/attempt/kind/f'{index:06d}.jsonl.gz'
                    complete = json.loads(chunk.with_suffix('.complete.json').read_text())
                    if file_hash(chunk) != complete['sha256']:
                        raise ValueError('log chunk checksum mismatch')
                    with gzip.open(chunk, 'rt') as source:
                        for line in source:
                            json.loads(line)
                            output.write(line)
