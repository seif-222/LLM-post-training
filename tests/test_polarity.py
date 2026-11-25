"""Local scientific, reproducibility and recovery acceptance fixtures; no remote calls."""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
import train_grpo as tg
from rl.campaign import (canonical, request_identity, training_requests, evaluation_requests,
                         isolated_rng, config_hash, Ledger, validate_authorization, worker_deadline)
from rl.checkpointing import atomic_json, validate, ChunkLog, file_hash
from rl.evaluation_io import evaluate_manifest, validate_records
from rl.grpo import group_advantages, mask_advantages, token_weights, grpo_loss


def model_fixture():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen2Config
    torch.manual_seed(17)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return get_peft_model(AutoModelForCausalLM.from_config(config),
                         LoraConfig(r=4, lora_alpha=8, target_modules='all-linear', lora_dropout=0.0))


def groups_fixture():
    return [{'idx': 0, 'seqs': [{'prompt_ids': torch.tensor([1, 2]),
                               'comp_ids': torch.tensor([3, 4, 5][:1+i%3]), 'length': 1+i%3,
                               'reward': float(i < 3), 'text': '', 'truncated': i % 2 == 0}
                              for i in range(8)]}]


def setup_model():
    model = model_fixture()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s+1)/10))
    return model, opt, sched


@pytest.mark.parametrize('correct', [0, 1, 4, 7, 8])
def test_advantages(correct):
    rewards = torch.tensor([1.] * correct + [0.] * (8-correct))
    adv = group_advantages(rewards, 8)
    expected = (rewards-rewards.mean())/(rewards.std(correction=0)+1e-4)
    torch.testing.assert_close(adv, expected, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(adv).all()
    if correct in (0, 8):
        assert not adv.any()
    assert torch.equal(mask_advantages(adv, 'full'), mask_advantages(adv, 'positive') + mask_advantages(adv, 'negative'))


def test_policy_decomposition_kl_and_microbatches():
    torch.manual_seed(5)
    logp = torch.randn(8, 4, requires_grad=True)
    mask = torch.arange(4).unsqueeze(0) < torch.tensor([1, 2, 3, 4]*2).unsqueeze(1)
    weights = token_weights(mask, 'seq')
    raw = group_advantages(torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.]), 8)
    grads, kls = {}, []
    for mode in ('full', 'positive', 'negative'):
        adv = mask_advantages(raw, mode)
        loss, info = grpo_loss(logp, logp.detach(), logp.detach()+.2, adv, weights, beta=.02)
        kls.append(info['kl_part'])
        policy, _ = grpo_loss(logp, logp.detach(), None, adv, weights)
        grads[mode] = torch.autograd.grad(policy, logp, retain_graph=True)[0]
        parts = []
        for sl in (slice(0, 3), slice(3, 8)):
            w = token_weights(mask[sl], 'seq', n_seqs=8)
            part, _ = grpo_loss(logp[sl], logp.detach()[sl], logp.detach()[sl]+.2, adv[sl], w, beta=.02)
            parts.append(part)
        torch.testing.assert_close(sum(parts), loss)
    torch.testing.assert_close(grads['full'], grads['positive']+grads['negative'])
    assert kls[0] == kls[1] == kls[2]
    assert weights.sum().item() == pytest.approx(1.)


def test_train_step_diagnostics_does_not_change_update():
    torch.set_num_threads(2)
    outputs = []
    for enabled in (False, True):
        model, opt, sched = setup_model()
        args = tg.parse_args(['--run', 'fixture', '--lr', '0.0001', '--micro-bs', '3', '--diagnostic-updates', '1' if enabled else ''])
        args.current_update = 1
        initial = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        stats = tg.train_step(model, opt, sched, groups_fixture(), args, 'cpu', 0)
        final = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        assert any(not torch.equal(a, b) for a,b in zip(initial, final))
        assert stats['lr'] == pytest.approx(1e-5)
        assert stats['trained_answers'] == 8 and stats['trained_tokens'] == 15
        assert stats['parameter_rms_change'] > 0
        outputs.append((final, opt.state_dict(), torch.get_rng_state(), stats))
    for a,b in zip(outputs[0][0], outputs[1][0]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(outputs[0][2], outputs[1][2])
    for key in outputs[0][1]['state']:
        for name in outputs[0][1]['state'][key]:
            torch.testing.assert_close(outputs[0][1]['state'][key][name], outputs[1][1]['state'][key][name], atol=0, rtol=0)
    assert outputs[1][3]['policy_kl_alignment'] is None  # initial effective adapter zero => zero KL gradient


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_before_optimizer(bad, monkeypatch):
    model, opt, sched = setup_model()
    args = tg.parse_args(['--run', 'fixture'])
    groups = groups_fixture()
    groups[0]['seqs'][0]['reward'] = bad
    called = []
    monkeypatch.setattr(opt, 'step', lambda: called.append(True))
    with pytest.raises(FloatingPointError):
        tg.train_step(model, opt, sched, groups, args, 'cpu', 0)
    assert not called


def test_request_seeds_and_rng_isolation():
    expected = {'version': 1, 'namespace': 'evaluation', 'evaluation_seed': 1234, 'dataset': 'test', 'question_id': 7}
    request = evaluation_requests('test', [7])[0]
    assert request['identity'] == expected
    assert request['seed'] == int.from_bytes(hashlib.sha256(canonical(expected)).digest()[:4], 'big') & 0x7fffffff
    requests = training_requests(0, 2, 0, [7, 7])
    assert requests[0]['seed'] != requests[1]['seed']
    assert requests != training_requests(0, 3, 0, [7, 7])
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    with isolated_rng(9):
        random.random(); np.random.random(); torch.rand(3)
    got = (random.random(), np.random.random(), torch.rand(3))
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    assert got[0] == random.random() and got[1] == np.random.random()
    assert torch.equal(got[2], torch.rand(3))


def test_checkpoint_resume_after_real_update_and_rejection(tmp_path):
    model, opt, sched = setup_model()
    args = tg.parse_args(['--run', 'fixture', '--diagnostic-updates', ''])
    stream, reward_gen = tg.QuestionStream(list(range(20)), 4), torch.Generator().manual_seed(7)
    stream.next(5)
    tg.train_step(model, opt, sched, groups_fixture(), args, 'cpu', 0)
    path = str(tmp_path/'ckpt')
    tg.save_checkpoint(path, model, opt, sched, stream, reward_gen, 1, {'config_hash': config_hash(args), 'attempt': 2})
    expected_questions = stream.next(6)
    expected = tg.train_step(model, opt, sched, groups_fixture(), args, 'cpu', 0)
    expected_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    fresh, fresh_opt, fresh_sched = setup_model()
    fresh_stream, fresh_gen = tg.QuestionStream(list(range(20)), 0), torch.Generator()
    state = tg.load_checkpoint(path, fresh, fresh_opt, fresh_sched, fresh_stream, fresh_gen, config_hash(args))
    assert state['step'] == 1 and state['attempt'] == 2
    assert fresh_stream.next(6) == expected_questions
    actual = tg.train_step(fresh, fresh_opt, fresh_sched, groups_fixture(), args, 'cpu', 0)
    assert actual['lr'] == expected['lr']
    for expected_p, p in zip(expected_params, [p for p in fresh.parameters() if p.requires_grad]):
        torch.testing.assert_close(expected_p, p, atol=0, rtol=0)
    with pytest.raises(ValueError, match='configuration'):
        tg.load_checkpoint(path, fresh, fresh_opt, fresh_sched, fresh_stream, fresh_gen, 'wrong')
    pointer = json.loads((Path(path)/'latest.json').read_text())
    state_path = Path(path)/pointer['generation']/'state.pt'
    state_path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        tg.load_checkpoint(path, fresh, fresh_opt, fresh_sched, fresh_stream, fresh_gen)
    with pytest.raises(FileNotFoundError):
        tg.load_checkpoint(str(tmp_path/'missing'), fresh, fresh_opt, fresh_sched, fresh_stream, fresh_gen)


def test_checkpoint_failed_commit_keeps_previous(tmp_path):
    model, opt, sched = setup_model()
    stream, gen = tg.QuestionStream([1,2], 0), torch.Generator()
    path = str(tmp_path/'ckpt')
    tg.save_checkpoint(path, model, opt, sched, stream, gen, 0, {})
    pointer = (Path(path)/'latest.json').read_text()
    calls = []
    def fail():
        calls.append(1)
        raise OSError('simulated volume failure')
    with pytest.raises(OSError):
        tg.save_checkpoint(path, model, opt, sched, stream, gen, 25, {}, commit=fail)
    assert len(calls) == 2 and (Path(path)/'latest.json').read_text() == pointer
    assert tg.load_checkpoint(path, model, opt, sched, stream, gen)['step'] == 0


def test_chunk_log_and_retention(tmp_path):
    log = ChunkLog(tmp_path/'attempt-a')
    log.write('{"step": 1}\n'); log.flush()
    log.write('{"step": 2}\n'); log.close()
    assert len(list((tmp_path/'attempt-a').glob('*.gz'))) == 2
    with pytest.raises(FileExistsError):
        ChunkLog(tmp_path/'attempt-a')
    model, opt, sched = setup_model()
    stream, gen = tg.QuestionStream([1, 2], 0), torch.Generator()
    for step in (0, 25, 50):
        tg.save_checkpoint(str(tmp_path/'ckpt'), model, opt, sched, stream, gen, step, {})
    assert len(list((tmp_path/'ckpt').glob('update-*'))) == 2


class FakeEval:
    def __init__(self):
        self.calls = []
    def greedy(self, questions):
        self.calls.append([q['id'] for q in questions])
        return [{'text': '\\boxed{1}', 'length': 1, 'token_ids': [9], 'finish_reason': 'stop', 'truncated': False} for _ in questions]


def fixture_manifest(n=65):
    return {'dataset': 'fixture', 'revision': 'fixture-v1', 'questions': [{'id': i, 'question': str(i), 'answer': 1.} for i in range(n)]}


def test_evaluation_resume_exact_ids_and_corruption(tmp_path):
    engine = FakeEval()
    first = evaluate_manifest(engine, fixture_manifest(), tmp_path, 1, sampled=False, model_identity='base')
    assert len(engine.calls) == 3
    second = evaluate_manifest(engine, fixture_manifest(), tmp_path, 1, sampled=False, model_identity='base')
    assert first == second and len(engine.calls) == 3
    (tmp_path/'chunks/0002.sha256').unlink()
    evaluate_manifest(engine, fixture_manifest(), tmp_path, 1, sampled=False, model_identity='base')
    assert len(engine.calls) == 4 and engine.calls[-1] == [64]
    records = json.loads((tmp_path/'chunks/0000.json').read_text())
    for bad in (records[:-1], records+[records[0]], [{**records[0], 'id': 999}]+records[1:]):
        with pytest.raises(ValueError, match='IDs'):
            validate_records(bad, fixture_manifest()['questions'][:32], 1)
    records[0]['samples'] = []
    with pytest.raises(ValueError, match='sample count'):
        validate_records(records, fixture_manifest()['questions'][:32], 1)
    (tmp_path/'chunks/0000.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        evaluate_manifest(engine, fixture_manifest(), tmp_path, 1, sampled=False, model_identity='base')


def test_ledger_budget_retry_deadline_disconnect():
    saved = []
    ledger = Ledger(3., 200., lambda state: saved.append(copy.deepcopy(state)))
    for name in ('a', 'b', 'c'):
        ledger.reserve(name, 1., 50., 100.)
        ledger.dispatched(name, 'call-'+name)
    with pytest.raises(ValueError):
        ledger.reserve('d', 1., 50., 100.)
    resumed = Ledger(3., 200., lambda state: None, state=copy.deepcopy(saved[-1]))
    with pytest.raises(ValueError, match='ownership'):
        resumed.reserve('a', 1., 50., 100.)
    resumed.finish('a', .2, False)
    with pytest.raises(ValueError, match='budget'):
        resumed.reserve('a', 1., 50., 100., prior_dead=True, transient=True)
    resumed.reserve('a', .8, 50., 100., prior_dead=True, transient=True)
    resumed.dispatched('a', 'retry-a'); resumed.finish('a', .1, False)
    with pytest.raises(ValueError, match='retry'):
        resumed.reserve('a', .1, 5., 100., prior_dead=True, transient=True)
    assert worker_deadline(100, 1, 2, 3600, 4000) == 1900
    with pytest.raises(ValueError, match='authorization'):
        validate_authorization({'launch': False})


def test_training_loop_empty_attempt_and_real_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rl.engine import pack_completions
    class Tokenizer:
        pad_token_id = 0
        padding_side = 'left'
        def batch_decode(self, ids, **kwargs):
            return ['unused']*len(ids)
    class Engine:
        def sample(self, problems, k, temperature, requests=None):
            # Attempt 1 is empty; later requests have mixed rewards. Uses identities, so resume is stable.
            ro = pack_completions([[1,2] for _ in problems], [[3,4,9] for _ in range(k*len(problems))],
                                  [[0.,0.,0.] for _ in range(k*len(problems))], [True]*(k*len(problems)),
                                  k, Tokenizer(), [9])
            del ro['sample_logp']
            ro['texts'] = ['\\boxed{1}' if requests[i//k]['identity']['attempt'] > 1 and i%k < 3 else '\\boxed{0}' for i in range(k*len(problems))]
            return ro
        def sync(self, step): pass
        def release(self): pass
    monkeypatch.setattr(tg.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(tg.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:model_fixture())
    monkeypatch.setattr(tg,'get_peft_model',lambda model,config:model)
    monkeypatch.setattr(tg,'make_engine',lambda *a,**k:Engine())
    monkeypatch.setattr(tg,'load_gsm8k',lambda *a,**k:[{'id':i,'question':str(i),'answer':1.} for i in range(20)])
    common = ['--run','integrated','--out-root',str(tmp_path),'--questions','1','--k','8','--oversample','0',
              '--refill-rounds','0','--eval-every','0','--passk-n','0','--ckpt-every','1','--no-grad-ckpt',
              '--lr','0.0001','--diagnostic-updates','1,2']
    tg.main(common+['--steps','1'],_test_device='cpu')
    root = tmp_path/'integrated'
    rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['type']=='train']
    assert len(updates)==1 and updates[0]['rollout_attempt']==2 and updates[0]['step']==1
    assert updates[0]['lr']==pytest.approx(1e-5)
    assert len([r for r in rows if r['type']=='empty_attempt'])==1
    with (root/'metrics.jsonl').open('a') as f:
        f.write('{"step": 999, "type":"uncommitted"}\n')
    tg.main(common+['--steps','2','--resume'],_test_device='cpu')
    rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['type']=='train']
    assert [r['step'] for r in updates]==[1,2]
    assert updates[-1]['rollout_attempt']==3 and updates[-1]['lr']==pytest.approx(2e-5)
    assert not any(r['step']==999 for r in rows)
    assert any('999' in p.read_text() for p in root.glob('metrics.jsonl.attempt-*'))


def test_campaign_comparisons_align_ids_and_do_not_pool_seeds(tmp_path):
    from scripts.compare import campaign_compare, save_campaign_comparison
    from rl.campaign import RUNS
    def row(i,c):
        return {'id':i,'question':str(i),'answer':1.,'n_correct':c,
                'samples':[{'correct_strict':j<c} for j in range(4)],'template_id':i//2}
    models={name:[row(i,(i+j)%5) for i in range(10)] for j,name in enumerate(['original',*RUNS])}
    models['positive_s1'].reverse()
    result=campaign_compare(models)
    seed0=next(r['score'] for r in result['table'] if r['model']=='positive_s0' and r['k']==2)
    seed1=next(r['score'] for r in result['table'] if r['model']=='positive_s1' and r['k']==2)
    mean=next(r['score'] for r in result['table'] if r['model']=='positive_mean' and r['k']==2)
    assert mean==pytest.approx((seed0+seed1)/2)
    save_campaign_comparison(result,tmp_path/'comparison')
    assert (tmp_path/'comparison/passk.png').exists()
    symbolic=campaign_compare(models,symbolic=True)
    assert symbolic['bootstrap_unit']=='template'
    models['full_s0'].append(models['full_s0'][0])
    with pytest.raises(ValueError,match='IDs'):
        campaign_compare(models)


def test_dry_run_and_modal_import_do_not_access_remote(monkeypatch):
    import importlib.util
    import modal
    from scripts.run_campaign import preparation_manifest,dry_run
    def forbidden(*a,**k):
        raise AssertionError('remote/image operation during dry run or import')
    monkeypatch.delenv('LM_RL_LAUNCH_MANIFEST',raising=False)
    monkeypatch.setattr(modal.Image,'debian_slim',forbidden)
    monkeypatch.setattr(modal.Function,'from_name',forbidden)
    spec=importlib.util.spec_from_file_location('polarity_modal_import',Path(tg.__file__).parent/'modal_app.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.app is None
    result=dry_run(preparation_manifest('fixture'))
    assert result['remote_calls']==0 and not result['launch_allowed']


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_gradient_stops_before_step_and_bad_state_not_published(tmp_path, monkeypatch, bad):
    model,opt,sched=setup_model()
    args=tg.parse_args(['--run','fixture','--diagnostic-updates',''])
    param=next(p for p in model.parameters() if p.requires_grad)
    hook=param.register_hook(lambda grad: torch.full_like(grad,bad))
    called=[]
    monkeypatch.setattr(opt,'step',lambda:called.append(True))
    with pytest.raises(FloatingPointError,match='gradient'):
        tg.train_step(model,opt,sched,groups_fixture(),args,'cpu',0)
    assert not called
    hook.remove()
    with torch.no_grad(): param.flatten()[0]=bad
    with pytest.raises(FloatingPointError,match='parameters'):
        tg.save_checkpoint(str(tmp_path/'bad'),model,opt,sched,tg.QuestionStream([1],0),torch.Generator(),0,{})
    assert not (tmp_path/'bad/latest.json').exists()


def test_log_lineage_excludes_failed_tail(tmp_path):
    from rl.checkpointing import restore_logs
    root=tmp_path
    chunks=ChunkLog(root/'attempts/a/metrics')
    chunks.write('{"step":1}\n');chunks.flush()
    chunks.write('{"step":999}\n');chunks.flush()
    restore_logs(root,{'a':{'metrics':1}})
    assert [json.loads(l)['step'] for l in (root/'metrics.jsonl').read_text().splitlines()]==[1]
    assert (root/'attempts/a/metrics/000001.jsonl.gz').exists()


def test_sampled_evaluation_chunk_resume_and_sample_counts(tmp_path):
    class SampleEngine:
        calls=0
        def sample(self,questions,k,temperature,requests=None):
            from rl.engine import pack_completions
            class Tok:
                pad_token_id=9
                def batch_decode(self,ids,**kwargs):return ['\\boxed{1}']*len(ids)
            self.calls+=1
            return pack_completions([[1] for q in questions],[[3,9] for _ in range(len(questions)*k)],
                                    [[0.,0.] for _ in range(len(questions)*k)],[True]*(len(questions)*k),k,Tok(),[9])
    engine=SampleEngine()
    result=evaluate_manifest(engine,fixture_manifest(),tmp_path,4,model_identity='fixture')
    assert result['passk']=={'pass@1':1.,'pass@2':1.,'pass@4':1.}
    evaluate_manifest(engine,fixture_manifest(),tmp_path,4,model_identity='fixture')
    assert engine.calls==3
    with pytest.raises(ValueError,match='configuration'):
        evaluate_manifest(engine,fixture_manifest(),tmp_path,2,model_identity='fixture')


def test_measured_uniform_cost_fallback_and_caps():
    from rl.campaign import choose_evaluation_k, validate_main_jobs, RUNS
    from scripts.run_campaign import preparation_manifest
    assert choose_evaluation_k(30,2.5,10,10,5)==64
    assert choose_evaluation_k(30,2.5,15,10,5)==32
    with pytest.raises(ValueError,match='neither'):
        choose_evaluation_k(15,2.5,10,10,5)
    with pytest.raises(ValueError,match='unresolved'):
        choose_evaluation_k(None,2.5,10,10,5)
    with pytest.raises(ValueError,match='thirteen'):
        validate_main_jobs(preparation_manifest('fixture'))
    manifest={'launch':True,'authorization':'synthetic test only','phase':'main','budget_usd':20,
              'deadline_utc_epoch':20000,'credit_expiry_utc_epoch':24000,'verified_credit_usd':30,
              'preflight_spent_usd':2.5,'initial_verified_credit_usd':30.,'k':64,'runs':list(RUNS)}
    validate_authorization(manifest,now=100)
    for key,value in [('k',None),('deadline_utc_epoch',23000),('budget_usd',29)]:
        bad={**manifest,key:value}
        with pytest.raises(ValueError):validate_authorization(bad,now=100)


def test_concurrent_reservations_are_persisted_before_dispatch():
    # A sole controller serializes simultaneous requests, including across a disconnection.
    states=[]
    ledger=Ledger(2.,100.,lambda state:states.append(copy.deepcopy(state)))
    ledger.reserve('a',1.5,10.,0.)
    recreated=Ledger(2.,100.,lambda state:None,state=copy.deepcopy(states[-1]))
    with pytest.raises(ValueError,match='budget'):
        recreated.reserve('b',1.,10.,0.)
    with pytest.raises(ValueError,match='ownership'):
        recreated.reserve('a',1.,10.,0.)
    assert states[-1]['jobs']['a']['status']=='reserved'


def test_hard_worker_deadline_cancels_timer():
    import threading
    import time
    from rl.campaign import hard_deadline
    fired=threading.Event()
    with hard_deadline(time.time()+.01,terminate=fired.set):
        assert fired.wait(.2)
    cancelled=threading.Event()
    with hard_deadline(time.time()+.05,terminate=cancelled.set):
        pass
    assert not cancelled.wait(.1)


@pytest.mark.parametrize("transient_job", ["full_s0", "negative_s1-evaluation"])
def test_controller_waves_retry_and_duplicate_with_fake_modal(tmp_path, monkeypatch, transient_job):
    import importlib.util
    from types import SimpleNamespace
    from modal.types import InputStatus
    from scripts.run_campaign import preparation_manifest
    from rl.checkpointing import file_hash
    manifest=preparation_manifest('controller-fixture')
    manifest.update(launch=True,authorization='synthetic fixture only; fake Modal backend',
                    budget_usd=1.,verified_credit_usd=30.,initial_verified_credit_usd=30.,preflight_spent_usd=.2,k=64,
                    deadline_utc_epoch=__import__('time').time()+1000,
                    credit_expiry_utc_epoch=__import__('time').time()+3000,
                    controller_reservation_usd=.01,
                    source_hashes={'rl/campaign.py':file_hash(Path(tg.__file__).parent/'rl/campaign.py')},
                    dataset_manifests={'gsm8k':'synthetic','symbolic':'synthetic'})
    by_name={c['logical_run']:c for c in manifest['configs']}
    for wave in manifest['waves']:
        for name in wave:
            job={'logical_run':name,'kind':'train' if name in by_name else 'evaluation',
                 'allocation_usd':.01,'seconds':10,'price_per_hour':2.,'boundary_margin_seconds':1.}
            if name in by_name:job['argv']=by_name[name]['argv']
            manifest['jobs'].append(job)
    events=[]
    active=set()
    max_active=[0]
    owners={}
    class InternalFailure(Exception):pass
    class Image:
        @classmethod
        def debian_slim(cls,**kwargs):return cls()
        def __getattr__(self,name):return lambda *a,**k:self
    class Volume:
        @classmethod
        def from_name(cls,*a,**k):return cls()
        def commit(self):events.append(('commit',))
        def reload(self):pass
    class Owner:
        def put(self,key,value,skip_if_exists=False):
            if key in owners:return False
            owners[key]=value;return True
    class Call:
        def __init__(self,payload):
            self.payload=payload;self.object_id=f'fake-{payload["logical_run"]}-{payload["attempt"]}'
        def get(self,**kwargs):
            active.remove(self.object_id)
            if self.payload['logical_run']==transient_job and self.payload['attempt']==1:
                raise InternalFailure('simulated terminal infrastructure failure')
            return {'complete':True}
        def get_call_graph(self):return [SimpleNamespace(status=InputStatus.FAILURE,children=[])]
    class Function:
        def __init__(self,fn):self.raw=fn
        def with_options(self,**kwargs):return self
        def spawn(self,payload,approved):
            ledger=json.loads((tmp_path/'campaigns/controller-fixture/controller.json').read_text())
            assert ledger['jobs'][payload['logical_run']]['status']=='reserved'
            call=Call(payload);active.add(call.object_id);max_active[0]=max(max_active[0],len(active))
            events.append(('spawn',payload['logical_run'],payload['attempt']))
            return call
    class App:
        def __init__(self,*a,**k):self.functions={}
        def function(self,**kwargs):
            def wrap(fn):
                f=Function(fn);self.functions[kwargs['name']]=f;return f
            return wrap
        def get_dashboard_url(self):return 'https://example.invalid/simulated-modal'
    fake=SimpleNamespace(Image=Image,Volume=Volume,App=App,
                         Secret=SimpleNamespace(from_name=lambda *a,**k:None),
                         Dict=SimpleNamespace(from_name=lambda *a,**k:Owner()),
                         exception=SimpleNamespace(InternalFailure=InternalFailure))
    monkeypatch.delenv('LM_RL_LAUNCH_MANIFEST',raising=False)
    spec=importlib.util.spec_from_file_location('controller_fixture_module',Path(tg.__file__).parent/'modal_app.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'modal',fake);monkeypatch.setattr(module,'VOL',str(tmp_path))
    monkeypatch.setattr(module,'CODE',str(Path(tg.__file__).parent))
    app=module.create_campaign_app(manifest)
    result=app.functions['campaign_controller'].raw(manifest)
    assert result['complete'] and max_active[0]==3 and not active
    ledger=json.loads((tmp_path/'campaigns/controller-fixture/controller.json').read_text())
    assert len(ledger['jobs'])==13 and ledger['jobs'][transient_job]['attempt']==2
    assert all(j['status']=='done' and j['call_id'].startswith('fake-') for j in ledger['jobs'].values())
    assert len([e for e in events if e[0]=='spawn'])==14
    with pytest.raises(RuntimeError,match='ownership'):
        app.functions['campaign_controller'].raw(manifest)


def test_crash_during_checkpoint_pointer_publication(tmp_path,monkeypatch):
    from rl import checkpointing as cp
    model,opt,sched=setup_model()
    stream,gen=tg.QuestionStream([1,2],0),torch.Generator()
    path=str(tmp_path/'ckpt')
    tg.save_checkpoint(path,model,opt,sched,stream,gen,0,{})
    original=cp.atomic_json
    def crash(path,value):
        if Path(path).name=='latest.json':raise OSError('simulated crash before pointer publication')
        return original(path,value)
    monkeypatch.setattr(cp,'atomic_json',crash)
    with pytest.raises(OSError):tg.save_checkpoint(path,model,opt,sched,stream,gen,25,{})
    assert tg.load_checkpoint(path,model,opt,sched,stream,gen)['step']==0


def test_three_empty_attempts_stop_without_optimizer(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tg.AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(pad_token_id=0,padding_side='left'))
    monkeypatch.setattr(tg.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:model_fixture())
    monkeypatch.setattr(tg,'get_peft_model',lambda model,config:model)
    monkeypatch.setattr(tg,'load_gsm8k',lambda *a,**k:[{'id':0,'question':'fixture','answer':1.}])
    monkeypatch.setattr(tg,'make_engine',lambda *a,**k:SimpleNamespace(release=lambda:None))
    sample={'length':1,'correct':False,'formatted':False,'truncated':False}
    monkeypatch.setattr(tg,'sample_groups',lambda *a,**k:([],{'n_right':[0],'seqs':[sample],
                        'rewards':[0.],'rounds':1,'t_rounds':[0.],'generated_tokens':1,'generated_answers':1}))
    def must_not_train(*a,**k):raise AssertionError('empty attempt reached optimizer')
    monkeypatch.setattr(tg,'train_step',must_not_train)
    with pytest.raises(RuntimeError,match='no-signal'):
        tg.main(['--run','empty','--out-root',str(tmp_path),'--steps','1','--eval-every','0',
                 '--passk-n','0','--no-grad-ckpt'],_test_device='cpu')
    rows=[json.loads(line) for line in (tmp_path/'empty/metrics.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['type']=='empty_attempt'])==3
    assert not any(r['type']=='train' for r in rows)


def test_pinned_dataset_rejects_silent_offline_fallback(monkeypatch):
    import rl.data as data
    from types import SimpleNamespace
    monkeypatch.setattr(data,'load_dataset',lambda *a,**k:SimpleNamespace(cache_files=[{'filename':'/cache/wrong-revision/data.arrow'}]))
    with pytest.raises(ValueError,match='pinned revision'):
        data.load_gsm8k('test',revision='expected-revision')
