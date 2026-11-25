"""Explicit direct authorization: resume the six existing scientific runs safely."""
import json
import os
import time
from pathlib import Path


def validate_manifest(m, now=None):
    from rl.campaign import RUNS, polarity_configs
    now=time.time() if now is None else now
    if m.get('phase')!='direct_resume' or m.get('launch') is not True or not m.get('authorization'):
        raise ValueError('explicit resume authorization missing')
    if not 0<m['additional_budget_usd']<=30 or m['reserved_usd']>m['additional_budget_usd']:
        raise ValueError('resume budget cap')
    if not now<m['deadline_epoch']<=m['authorized_at_epoch']+4500:
        raise ValueError('resume absolute deadline')
    if m['reserved_usd']<6*4500*m['worker_max_usd_per_second']+m['overhead_usd'] or m['overhead_usd']<3:
        raise ValueError('incomplete whole-batch reservation')
    expected={j['logical_run']:j['argv'] for j in polarity_configs(m['scientific_campaign_id'])}
    if len(m['jobs'])!=6 or {j['logical_run'] for j in m['jobs']}!=set(RUNS):
        raise ValueError('exact six jobs required')
    for j in m['jobs']:
        if j['argv']!=expected[j['logical_run']]:raise ValueError('scientific settings changed')
        if j['checkpoint_step'] not in (25,50) or len(j['checkpoint_manifest_sha256'])!=64:
            raise ValueError('unverified origin checkpoint')
    if not m['source_hashes'] or not m['dataset_hashes']:raise ValueError('missing source/data identity')


def claim_owner(claims, key, call_id, identity):
    value={'call_id':call_id,'identity':identity}
    if claims.put(key,value,skip_if_exists=True):return False
    if claims[key]!=value:raise ValueError('another call owns this run')
    return True  # same platform-rescheduled invocation, never another live owner


def verify_checkpoint(run, m, job, origin=False):
    import torch
    from safetensors.torch import load_file
    from rl.checkpointing import validate,file_hash
    from rl.campaign import finite,config_hash
    from train_grpo import parse_args
    pointer=json.loads((run/'ckpt/latest.json').read_text())
    if Path(pointer['generation']).name!=pointer['generation']:raise ValueError('invalid generation')
    generation=run/'ckpt'/pointer['generation']
    checked=validate(generation,config_hash(parse_args(job['argv'])))
    if origin and (checked['step']!=job['checkpoint_step'] or file_hash(generation/'manifest.json')!=job['checkpoint_manifest_sha256']):
        raise ValueError('origin checkpoint changed')
    state=torch.load(generation/'state.pt',weights_only=False,map_location='cpu')
    scientific={n:h for n,h in m['source_hashes'].items() if n=='train_grpo.py' or n.startswith('rl/')}
    if state['source_hashes']!=scientific or state['step']!=checked['step']:
        raise ValueError('checkpoint scientific source mismatch')
    finite(state['opt']);finite(load_file(str(generation/'adapter_model.safetensors')))
    return {'step':checked['step'],'checkpoint':str(generation),'manifest_sha256':file_hash(generation/'manifest.json'),
            'checksums':checked['checksums'],'log_lineage':state['log_lineage']}


def prepare_run(m, job, root):
    """Copy immutable source artifacts once; leave the previous campaign untouched."""
    import shutil
    from rl.checkpointing import atomic_json,restore_logs
    source=Path(root)/'campaigns'/m['scientific_campaign_id']/'training'/job['logical_run']
    dest=Path(root)/'continuations'/m['continuation_id']/'training'/job['logical_run']
    marker=dest/'resume-origin.json'
    if marker.exists():
        if json.loads(marker.read_text())['manifest_sha256']!=job['checkpoint_manifest_sha256']:
            raise ValueError('resume origin differs')
        return verify_checkpoint(dest,m,job)
    original=verify_checkpoint(source,m,job,origin=True)
    # Copies have a separate output root, preserving original run files and tails.
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('ckpt','attempts','evals'):
        if (source/name).exists():shutil.copytree(source/name,dest/name,dirs_exist_ok=True)
    for name in ('config.json','metrics.jsonl','hard.jsonl','samples.jsonl.gz'):
        if (source/name).exists():shutil.copy2(source/name,dest/name)
    verify_checkpoint(dest,m,job,origin=True)
    restore_logs(dest,original['log_lineage'])
    atomic_json(marker,original)
    return original


def run_controller(m, state_path, persist, spawn, from_id, sleep=time.sleep, now=time.time):
    """Recover existing calls; cancellation is limited to cutoff or shared correctness failures."""
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['continuation_id']!=m['continuation_id']:raise ValueError('controller identity mismatch')
    else:
        state={'continuation_id':m['continuation_id'],'reserved_usd':m['reserved_usd'],
               'jobs':{j['logical_run']:{'status':'reserved','call_id':None} for j in m['jobs']}}
        persist(state)
    for job in m['jobs']:
        row=state['jobs'][job['logical_run']]
        if row['status']=='dispatching' and not row['call_id']:
            raise RuntimeError('uncertain dispatch ownership; reconciliation required')
        if row['status']=='reserved':
            row['status']='dispatching';persist(state)
            call=spawn(job)
            row.update(status='running',call_id=call.object_id);persist(state)
    pending={n:from_id(r['call_id']) for n,r in state['jobs'].items() if r['status']=='running'}
    while pending and now()<m['deadline_epoch']-45:
        for name,call in list(pending.items()):
            try:result=call.get(timeout=0)
            except TimeoutError:continue
            except Exception as exc:
                row=state['jobs'][name];row.update(status='failed',error=repr(exc));pending.pop(name)
                call.cancel(terminate_containers=True)
                if any(t in str(exc).lower() for t in ('nonfinite','source mismatch','fingerprint','sampler gap','configuration mismatch')):
                    for other,active in pending.items():
                        active.cancel(terminate_containers=True)
                        state['jobs'][other].update(status='cancelled-correctness')
                    persist(state);raise
            else:
                state['jobs'][name].update(status='done' if result['step']==200 else 'partial',result=result)
                pending.pop(name)
            persist(state)
        if pending:sleep(3)
    for name,call in pending.items():
        call.cancel(terminate_containers=True)
        state['jobs'][name]['status']='cancelled-deadline'
    persist(state)
    return state


def create_app(captured):
    import modal
    from modal_app import campaign_image,ROOT,CODE
    from rl.campaign import digest
    validate_manifest(captured)
    app=modal.App(captured['deployment_name'],image=campaign_image(captured['dataset_manifests']),include_source=False)
    vol=modal.Volume.from_name('lm-rl')
    identity=digest(captured)
    def enter(m):
        import sys
        os.chdir(CODE)
        if CODE not in sys.path:sys.path.insert(0,CODE)
        from rl.checkpointing import file_hash
        validate_manifest(m)
        if digest(m)!=identity:raise ValueError('manifest identity mismatch')
        for n,h in m['source_hashes'].items():
            if file_hash(Path(CODE)/n)!=h:raise ValueError('source fingerprint mismatch: '+n)
        for n,h in m['dataset_hashes'].items():
            if file_hash(Path(CODE)/'manifests'/f'{n}.json')!=h:raise ValueError('data fingerprint mismatch')
        vol.reload()
        return modal.Dict.from_name(m['deployment_name']+'-owners',create_if_missing=True)
    common=dict(volumes={'/vol':vol},retries=0,min_containers=0,buffer_containers=0,scaledown_window=2,
                single_use_containers=True,serialized=True,include_source=False)

    @app.function(name='resume_worker',gpu='L40S',cpu=(4,4),memory=(16384,16384),timeout=4500,
                  startup_timeout=300,max_containers=6,secrets=[modal.Secret.from_name('wandb-secret')],**common)
    def worker(job,m):
        from rl.campaign import hard_deadline
        from rl.checkpointing import atomic_json
        claims=enter(m)
        if job not in m['jobs']:raise ValueError('unapproved job')
        claim_owner(claims,job['logical_run'],modal.current_function_call_id(),identity)
        import train_grpo,wandb
        base=Path('/vol/continuations')/m['continuation_id']/'training'
        run=base/job['logical_run']
        original=verify_checkpoint(run,m,job)
        attempt_id=modal.current_input_id() if hasattr(modal,'current_input_id') else str(time.time_ns())
        attempts=claims.get('attempts-'+job['logical_run'],0)+1
        claims['attempts-'+job['logical_run']]=attempts
        if attempts>3:raise RuntimeError('bounded infrastructure retry limit')
        started=time.time()
        def commit():
            if wandb.run:atomic_json(run/'wandb-run.json',{'path':wandb.run.path,'url':wandb.run.url})
            vol.commit()
        def guard():
            if time.time()>=m['deadline_epoch']-180:raise TimeoutError('graceful budget deadline')
        partial=False
        try:
            with hard_deadline(m['deadline_epoch']-30):
                train_grpo.main(job['argv']+['--resume','--out-root',str(base)],on_checkpoint=commit,guard=guard)
        except TimeoutError:partial=True
        finally:
            if wandb.run:wandb.finish()
            vol.commit()
        result=verify_checkpoint(run,m,job)
        wandb_identity=json.loads((run/'wandb-run.json').read_text())
        wrun=wandb.Api(timeout=20).run(wandb_identity['path'])
        result.update(resumed_from=original['step'],duration_seconds=time.time()-started,partial=partial,
                      wandb_url=wrun.url,attempts=attempts,wandb_successful_update=wrun.summary.get('successful_update'))
        if result['step']==200 and result['wandb_successful_update']!=200:raise ValueError('W&B update200 not visible')
        atomic_json(run/'terminal.json',result);vol.commit();vol.reload()
        if verify_checkpoint(run,m,job)['manifest_sha256']!=result['manifest_sha256']:raise ValueError('readback mismatch')
        return result

    @app.function(name='resume_controller',cpu=(2,2),memory=(8192,8192),timeout=4500,startup_timeout=120,
                  nonpreemptible=True,max_containers=1,**common)
    def controller(m):
        from rl.checkpointing import atomic_json
        from rl.campaign import hard_deadline
        claims=enter(m)
        claim_owner(claims,'controller',modal.current_function_call_id(),identity)
        base=Path('/vol/continuations')/m['continuation_id']
        with hard_deadline(m['deadline_epoch']-15):
            # Preparation validates every origin before any GPU starts.
            prepared={j['logical_run']:prepare_run(m,j,'/vol') for j in m['jobs']}
            atomic_json(base/'prepared.json',prepared);vol.commit()
            def persist(state):atomic_json(base/'controller.json',state);vol.commit()
            return run_controller(m,base/'controller.json',persist,lambda j:worker.spawn(j,m),modal.FunctionCall.from_id)
    return app


app=None
if os.environ.get('LM_RL_RESUME_MANIFEST'):
    app=create_app(json.loads(Path(os.environ['LM_RL_RESUME_MANIFEST']).read_text()))
