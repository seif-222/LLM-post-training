"""Polarity campaign configuration, request identity and local controller state.

Importing this module never imports Modal or initiates a network request.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import random
import time
from pathlib import Path

MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
DATA_REVISION = '740312add88f781978c0658806c59bc2815b9866'
SEED_VERSION = 1
ARMS = ('full', 'positive', 'negative')
RUNS = tuple(f'{arm}_s{seed}' for seed in (0, 1) for arm in ARMS)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def request_identity(namespace, **fields):
    identity = {'version': SEED_VERSION, 'namespace': namespace, **fields}
    seed = int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:4], 'big') & 0x7fffffff
    return {'identity': identity, 'seed': seed}


def training_requests(seed, attempt, round_number, ids):
    counts, requests = {}, []
    for qid in ids:
        occurrence = counts.get(qid, 0)
        requests.append(request_identity('training', training_seed=seed, attempt=attempt,
                                        refill_round=round_number, question_id=qid, occurrence=occurrence))
        counts[qid] = occurrence + 1
    return requests


def evaluation_requests(dataset, ids):
    return [request_identity('evaluation', evaluation_seed=1234, dataset=dataset, question_id=qid) for qid in ids]


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_rng(seed=None):
    state = rng_state()
    try:
        if seed is not None:
            import numpy as np
            import torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        yield
    finally:
        restore_rng(state)


def finite(value, name='state'):
    import torch
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'nonfinite {name}')
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f'{name}.{key}')
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item, name)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f'nonfinite {name}')


def config_hash(args):
    # Paths, logging identity and an explicitly extended legacy run are not scientific settings.
    ignored = {'resume', 'run', 'out_root', 'wandb', 'wandb_project', 'price_per_hour'}
    if not getattr(args, 'campaign_id', None):
        ignored.add('steps')
    return digest({k: v for k, v in vars(args).items() if k not in ignored})


def polarity_configs(campaign_id):
    from train_grpo import parse_args
    configs = []
    for seed in (0, 1):
        for mode in ARMS:
            flags = ['--run', f'{mode}_s{seed}', '--campaign-id', campaign_id, '--seed', str(seed),
                     '--advantage-mode', mode, '--engine', 'vllm', '--no-grad-ckpt', '--wandb',
                     '--lr', '0.0001', '--steps', '200', '--oversample', '16', '--questions', '16',
                     '--k', '8', '--beta', '0.02', '--passk-n', '0', '--eval-n', '0', '--eval-every', '25',
                     '--ckpt-every', '25', '--micro-bs', '8', '--diagnostic-updates', '25,100,200',
                     '--max-attempts', '220', '--max-empty-attempts', '3', '--model-revision', MODEL_REVISION,
                     '--data-revision', DATA_REVISION]
            args = parse_args(flags)
            configs.append({'logical_run': args.run, 'argv': flags, 'resolved': vars(args), 'config_hash': config_hash(args),
                            'fixed_scientific_settings': {
                                'optimizer': {'name':'AdamW','betas':[0.9,0.99],'epsilon':1e-8,'weight_decay':0.},
                                'advantage_population_std':True,'advantage_epsilon':1e-4,'policy_rescaling':1.,
                                'reward_tolerance':1e-4,'cutoffs_scored_normally':True,
                                'generation': {'top_p':1.,'vllm_top_k':-1,'hf_top_k':0,'min_p':0.,
                                               'repetition_penalty':1.,'presence_penalty':0.,'frequency_penalty':0.},
                                'eos_ids':[151645,151643],'include_first_eos':True,
                                'lora_dropout':0.,'base_dtype':'bfloat16','lora_dtype':'float32',
                                'target_projections':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                                'layers':24,'gpu':'L40S','cpu_cores':4,'ram_gib':16,
                                'evaluation_vllm_memory_fraction':0.5,'evaluation_seed':1234,
                                'seed_derivation_version':1,'evaluation_chunk_questions':32}})
    return configs


def validate_authorization(manifest, now=None):
    """Fail before importing/building a remote app if any funding/approval field is unresolved."""
    now = time.time() if now is None else now
    if manifest.get('launch') is not True or not manifest.get('authorization'):
        raise ValueError('explicit launch authorization is required')
    for name in ('budget_usd', 'deadline_utc_epoch', 'credit_expiry_utc_epoch', 'verified_credit_usd'):
        v = manifest.get(name)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'unresolved {name}')
    margin = 120 if manifest.get('phase') == 'training_only' else 1800
    if manifest['deadline_utc_epoch'] <= now or manifest['deadline_utc_epoch'] > manifest['credit_expiry_utc_epoch'] - margin:
        raise ValueError('deadline must precede verified credit expiry by 30 minutes')
    if manifest.get('phase') == 'training_only':
        if manifest['budget_usd'] + manifest.get('prior_spent_usd',0) > min(25, manifest['verified_credit_usd'] - 5):
            raise ValueError('training-only exceeds protected budget')
        if manifest.get('max_gpu_jobs') != 6 or manifest.get('retries') != 0:
            raise ValueError('training-only requires six jobs and zero retries')
        if manifest.get('runs') != list(RUNS):
            raise ValueError('training-only requires exact six runs')
        if manifest['deadline_utc_epoch'] - now > 3600:
            raise ValueError('training-only exceeds one hour')
    elif manifest.get('phase') == 'main':
        if manifest.get('k') not in (32, 64):
            raise ValueError('main evaluation k is unresolved')
        if manifest.get('runs') != list(RUNS):
            raise ValueError('main requires all six prescribed logical runs')
        initial_credit = manifest.get('initial_verified_credit_usd')
        if not isinstance(initial_credit, (float,int)) or not math.isfinite(initial_credit) or initial_credit<=0:
            raise ValueError('initial preflight credit is unresolved')
        budget = min(30, initial_credit) - 4.5
        if manifest['budget_usd'] > manifest['verified_credit_usd'] - 4.5:
            raise ValueError('current remaining credit no longer funds campaign')
        if manifest['budget_usd'] + manifest.get('preflight_spent_usd', math.inf) > budget:
            raise ValueError('campaign exceeds protected credit budget')
        if manifest['deadline_utc_epoch'] - now > 6 * 3600:
            raise ValueError('main exceeds six hours')
    elif manifest.get('phase') == 'preflight':
        if manifest['budget_usd'] > 2.5 or manifest['deadline_utc_epoch'] - now > 5400:
            raise ValueError('preflight exceeds caps')
    else:
        raise ValueError('unrecognized phase')


class Ledger:
    """Single-controller ledger. Persist reservation BEFORE dispatch; uncertain calls stay owned.

    The caller provides the durable storage/commit callback. A restarted controller must
    reconcile every reserved/running job, never silently redispatch an uncertain reservation.
    """
    def __init__(self, budget, deadline, persist, max_jobs=3, state=None):
        self.budget, self.deadline, self.persist, self.max_jobs = budget, deadline, persist, max_jobs
        self.state = state or {'spent': 0.0, 'jobs': {}}

    def reserve(self, name, dollars, seconds, now, prior_dead=False, transient=False):
        jobs = self.state['jobs']
        prev = jobs.get(name)
        if prev and (not prior_dead or not transient or prev['attempt'] >= 2 or prev['status'] != 'failed'):
            raise ValueError('duplicate, uncertain ownership or retry not authorized')
        active = [j for j in jobs.values() if j['status'] in ('reserved', 'running')]
        if len(active) >= self.max_jobs:
            raise ValueError('concurrency cap')
        if not all(math.isfinite(v) and v > 0 for v in (dollars, seconds)):
            raise ValueError('invalid reservation')
        if now + seconds > self.deadline:
            raise ValueError('deadline cap')
        if self.state['spent'] + sum(j['reserved_usd'] for j in active) + dollars > self.budget:
            raise ValueError('budget cap')
        jobs[name] = {'status': 'reserved', 'attempt': 1 if prev is None else prev['attempt'] + 1,
                      'reserved_usd': dollars, 'seconds': seconds, 'dispatch_time': now,
                      'deadline': min(self.deadline, now + seconds), 'call_id': None,
                      'history': [] if prev is None else prev['history'] + [prev.copy()]}
        self.persist(self.state)
        return jobs[name].copy()

    def dispatched(self, name, call_id):
        j = self.state['jobs'][name]
        if j['status'] != 'reserved' or not call_id:
            raise ValueError('invalid dispatch transition')
        j.update(status='running', call_id=call_id)
        self.persist(self.state)

    def finish(self, name, actual_usd, success):
        j = self.state['jobs'][name]
        if j['status'] not in ('reserved', 'running'):
            raise ValueError('job already settled')
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise ValueError('invalid actual cost')
        self.state['spent'] += actual_usd
        j.update(status='done' if success else 'failed', actual_usd=actual_usd)
        self.persist(self.state)
        if actual_usd > j['reserved_usd']:
            raise RuntimeError('actual charges exceeded reservation; stop dispatch')


def worker_deadline(dispatch_time, allocation, price_per_hour, invocation_seconds, absolute_deadline):
    if allocation <= 0 or price_per_hour <= 0:
        raise ValueError('invalid worker allocation')
    return min(dispatch_time + allocation / price_per_hour * 3600,
               dispatch_time + invocation_seconds, absolute_deadline)


def assert_model_setup(model):
    """Campaign assertions; no alternative precision, targets or adapter initialization."""
    import torch
    expected = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    modules = [(n,m) for n,m in model.named_modules() if hasattr(m,'lora_A') and m.lora_A]
    if len(modules) != 24*7 or {n.split('.')[-1] for n,m in modules} != expected:
        raise ValueError('unexpected LoRA target modules')
    for name,module in modules:
        if module.r['default'] != 16 or module.lora_alpha['default'] != 32:
            raise ValueError('unexpected LoRA rank/alpha')
        if torch.count_nonzero(module.lora_B['default'].weight):
            raise ValueError('effective LoRA initialization is not zero')
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any('lora_' not in n or p.dtype != torch.float32 for n,p in trainable):
        raise ValueError('unexpected trainable parameters or LoRA dtype')
    if any(p.dtype != torch.bfloat16 for n,p in model.named_parameters() if not p.requires_grad and p.is_floating_point()):
        raise ValueError('frozen base is not bf16')
    count = sum(p.numel() for n,p in trainable)
    if not 8_700_000 <= count <= 8_900_000:
        raise ValueError('unexpected trainable parameter count')
    return {'trainable_parameters': count, 'targets': [n for n,m in modules],
            'lora_dtype': 'float32', 'base_dtype': 'bfloat16', 'effective_initial_adapter_zero': True}


def validate_main_jobs(manifest):
    """Check the fully costed dispatch plan before any remote lookup or image construction."""
    expected_waves = [['original-evaluation'], list(RUNS[:3]), list(RUNS[3:]),
                      [r+'-evaluation' for r in RUNS[:3]], [r+'-evaluation' for r in RUNS[3:]]]
    if manifest.get('waves') != expected_waves:
        raise ValueError('main wave order differs from prescribed campaign')
    jobs = manifest.get('jobs', [])
    names = [job['logical_run'] for job in jobs]
    expected_names = [name for wave in expected_waves for name in wave]
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise ValueError('main requires exactly thirteen training/evaluation jobs')
    configs = {r['logical_run']:r for r in polarity_configs(manifest['campaign_id'])}
    total = 0.
    for job in jobs:
        for key in ('allocation_usd','seconds','price_per_hour','boundary_margin_seconds'):
            value = job.get(key)
            if not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'unresolved job {key}')
        maximum = 4500 if job['kind']=='train' else 3600
        if job['seconds']>maximum or job['boundary_margin_seconds']>=job['seconds']:
            raise ValueError('job time cap or boundary margin invalid')
        if job['allocation_usd'] < job['seconds']/3600*job['price_per_hour']:
            raise ValueError('job reservation does not fund its complete time allocation')
        total+=job['allocation_usd']
        if job['kind']=='train':
            from train_grpo import parse_args
            got=parse_args(job['argv'])
            expected=configs[job['logical_run']]
            if vars(got)!=expected['resolved']:
                raise ValueError('training configuration differs from prescribed scientific settings')
        elif job['kind']=='evaluation':
            if job['logical_run'] not in [n+'-evaluation' for n in ('original',*RUNS)]:
                raise ValueError('unknown evaluation target')
        else:
            raise ValueError('unknown job kind')
    if total+manifest.get('controller_reservation_usd',0)>manifest['budget_usd']:
        raise ValueError('job and controller reservations exceed campaign budget')
    if manifest.get('controller_reservation_usd',0)<=0:
        raise ValueError('controller CPU cost reservation is unresolved')
    return True


def choose_evaluation_k(verified_credit, preflight_spent, fixed_cost, sampled_cost64, sampled_cost32):
    """Costs already include startup/save/greedy/diagnostics, resources and 25% compute margin.

    Two independently costed complete packages are required; no speed scaling is assumed.
    """
    values = (verified_credit, preflight_spent, fixed_cost, sampled_cost64, sampled_cost32)
    if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('unresolved measured costs/credit')
    available = min(30,verified_credit)-4.5
    for k, sampled in ((64,sampled_cost64),(32,sampled_cost32)):
        if preflight_spent+fixed_cost+sampled<=available:
            return k
    raise ValueError('neither complete evaluation package fits')


@contextmanager
def hard_deadline(deadline, terminate=None):
    """Worker-local last resort if a backend hangs past its funded absolute deadline.

    Normal boundary guards save a checkpoint earlier; this timer preserves the last
    committed state by terminating the stuck process instead of continuing to bill.
    """
    import os
    import threading
    timer = threading.Timer(max(0, deadline-time.time()), terminate or (lambda: os._exit(124)))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def validate_training_jobs(manifest):
    validate_authorization(manifest)
    if manifest.get('phase') != 'training_only' or manifest.get('waves') != [list(RUNS)]:
        raise ValueError('exactly one six-job training wave required')
    expected = {c['logical_run']: c for c in polarity_configs(manifest['campaign_id'])}
    jobs = manifest.get('jobs', [])
    if len(jobs) != 6 or {j['logical_run'] for j in jobs} != set(RUNS):
        raise ValueError('exactly six training jobs required')
    total = manifest.get('overhead_reservation_usd', 0)
    if total < 3.5:
        raise ValueError('acceptance/build/controller/shutdown reserve missing')
    for job in jobs:
        if job.get('kind') != 'train' or job.get('argv') != expected[job['logical_run']]['argv']:
            raise ValueError('wrong scientific training configuration')
        if job.get('allocation_usd',0) <= 0:
            raise ValueError('missing job reservation')
        total += job['allocation_usd']
    if total > manifest['budget_usd']:
        raise ValueError('aggregate reservations exceed cap')
    if not manifest.get('source_hashes') or not manifest.get('dataset_hashes'):
        raise ValueError('source/data fingerprints missing')
    return True


def reserve_training_batch(manifest, persist, now=None):
    validate_training_jobs(manifest)
    now = time.time() if now is None else now
    if now >= manifest['deadline_utc_epoch'] - 180:
        raise ValueError('graceful cutoff reached')
    state = {'campaign_id':manifest['campaign_id'], 'reserved_usd':sum(j['allocation_usd'] for j in manifest['jobs']) + manifest['overhead_reservation_usd'], 'prior_spent_usd':manifest.get('prior_spent_usd',0), 'aggregate_ceiling_usd':25,
             'jobs':{j['logical_run']:{'status':'reserved','attempt':1,'call_id':None,'dispatch_time':now,**j} for j in manifest['jobs']}}
    persist(state)
    return state
