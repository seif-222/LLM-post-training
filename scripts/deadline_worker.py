"""Narrow, explicitly authorized training-only deadline campaign. Import is inert."""
import json
import os
import time
from pathlib import Path


def verify_identity(approved, captured, code):
    from rl.campaign import digest
    from rl.checkpointing import file_hash
    if digest(approved) != digest(captured):
        raise ValueError('submitted manifest differs from deployed identity')
    for name, expected in approved['source_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(Path(code)/name) != expected:
            raise ValueError('remote source fingerprint mismatch: '+name)
    for name, expected in approved['dataset_hashes'].items():
        if file_hash(Path(code)/'manifests'/f'{name}.json') != expected:
            raise ValueError('remote dataset fingerprint mismatch: '+name)


def checkpoint_record(run_dir):
    from rl.checkpointing import validate, file_hash
    pointer = json.loads((run_dir/'ckpt/latest.json').read_text())
    generation = run_dir/'ckpt'/pointer['generation']
    checked = validate(generation)
    return {'step': checked['step'], 'checkpoint': str(generation),
            'manifest_sha256': file_hash(generation/'manifest.json'), 'checksums': checked['checksums']}


def run_training(approved, job, vol, root, smoke=False):
    import train_grpo
    from rl.campaign import hard_deadline
    argv = list(job['argv'])
    out = Path(root)/'campaigns'/approved['campaign_id']/('acceptance' if smoke else 'training')
    if smoke:
        for key,value in {'--steps':'2','--eval-every':'0','--ckpt-every':'2','--diagnostic-updates':'2'}.items():
            argv[argv.index(key)+1] = value
    argv += ['--out-root',str(out)]
    stop = min(approved['deadline_utc_epoch']-30, job.get('stop_at',float('inf')))
    def guard():
        if time.time() >= stop-(30 if smoke else 150):
            raise TimeoutError('graceful boundary stop')
    started = time.time()
    run_dir = out/job['logical_run']
    def commit_checkpoint():
        import wandb
        from rl.checkpointing import atomic_json
        if wandb.run:
            atomic_json(run_dir/'wandb-run.json',{'path':wandb.run.path,'url':wandb.run.url})
        vol.commit()
    partial = False
    try:
        with hard_deadline(stop):
            train_grpo.main(argv, on_checkpoint=commit_checkpoint, guard=guard)
    except TimeoutError:
        partial = True
    finally:
        import wandb
        if wandb.run:
            wandb.finish()
        vol.commit()
    run_dir = out/job['logical_run']
    result = checkpoint_record(run_dir)
    records = [json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()]
    train = [r for r in records if r['type']=='train' and r.get('parameter_rms_change') is not None]
    import torch
    from rl.campaign import finite
    state = torch.load(Path(result['checkpoint'])/'state.pt',map_location='cpu',weights_only=False)
    finite(state['opt'])
    cfg = json.loads((run_dir/'config.json').read_text())
    import wandb
    wandb_identity=json.loads((run_dir/'wandb-run.json').read_text())
    wrun = wandb.Api(timeout=15).run(wandb_identity['path'])
    result.update(duration_seconds=time.time()-started,partial=partial,wandb_url=wrun.url,
                  train_records=len(train), startup=[r for r in records if r['type']=='startup'],
                  max_update_seconds=max([r.get('t',r.get('t_gen',0)+r.get('t_train',0)) for r in train] or [0]))
    if smoke:
        if result['step'] != 2 or len(train)!=2 or not all(r['parameter_rms_change']>0 for r in train):
            raise ValueError('smoke requires two finite nonzero optimizer updates')
        finite(train)
    from rl.checkpointing import atomic_json
    atomic_json(run_dir/'terminal.json',result)
    vol.commit()
    vol.reload()
    if checkpoint_record(run_dir)['manifest_sha256'] != result['manifest_sha256']:
        raise ValueError('checkpoint readback mismatch')
    return result



def verify_saved_smoke(approved,vol,root):
    import torch,wandb
    from safetensors.torch import load_file
    from rl.campaign import finite
    from rl.checkpointing import file_hash
    proof=approved['verified_smoke']
    previous=Path(root)/'campaigns'/proof['campaign_id']/'acceptance'
    previous_manifest=json.loads((previous/'smoke-input.json').read_text())['manifest']
    for name in approved['source_hashes']:
        if name.startswith('rl/') or name in ('train_grpo.py','tests/test_gpu_vllm.py'):
            if approved['source_hashes'][name]!=previous_manifest['source_hashes'][name]:
                raise ValueError('changed model/trainer/test source invalidates smoke')
    if approved['dataset_hashes']!=previous_manifest['dataset_hashes']:
        raise ValueError('changed smoke data')
    if file_hash(previous/'smoke-log.txt')!=proof['log_sha256']:
        raise ValueError('smoke log checksum mismatch')
    run_dir=previous/'full_s0'
    result=checkpoint_record(run_dir)
    if result['step']!=2:raise ValueError('smoke not at update2')
    generation=Path(result['checkpoint'])
    finite(load_file(str(generation/'adapter_model.safetensors')))
    state=torch.load(generation/'state.pt',map_location='cpu',weights_only=False)
    finite(state['opt'])
    cfg=json.loads((run_dir/'config.json').read_text())
    from train_grpo import parse_args
    argv=list(previous_manifest['jobs'][0]['argv'])
    for key,value in {'--steps':'2','--eval-every':'0','--ckpt-every':'2','--diagnostic-updates':'2'}.items():
        argv[argv.index(key)+1]=value
    argv+=['--out-root',str(previous)]
    if cfg!=vars(parse_args(argv)):raise ValueError('smoke config differs')
    for name,sha in state['source_hashes'].items():
        if approved['source_hashes'][name]!=sha:raise ValueError('smoke checkpoint source mismatch')
    rows=[json.loads(x) for x in (run_dir/'metrics.jsonl').read_text().splitlines()]
    trained=[r for r in rows if r['type']=='train' and r.get('parameter_rms_change') is not None]
    if len(trained)!=2 or not all(r['parameter_rms_change']>0 for r in trained):raise ValueError('nonzero smoke updates missing')
    finite(trained)
    if proof['wandb_path'].split('/')[-1]!=state['wandb_id']:raise ValueError('W&B identity mismatch')
    wrun=wandb.Api(timeout=15).run(proof['wandb_path'])
    if wrun.config.get('campaign_id')!=proof['campaign_id'] or wrun.config.get('run')!='full_s0':raise ValueError('wrong W&B run')
    if wrun.summary.get('successful_update')!=2:raise ValueError('W&B update2 not visible')
    result.update(wandb_url=wrun.url,train_records=2,partial=False,
                  max_update_seconds=max(r['t_gen']+r['t_train'] for r in trained),
                  startup=[r for r in rows if r['type']=='startup'],verified_existing_smoke=True)
    vol.reload()
    if checkpoint_record(run_dir)['manifest_sha256']!=result['manifest_sha256']:raise ValueError('smoke readback mismatch')
    return result


def install(app, vol, captured, code, root):
    import modal
    from rl.campaign import digest
    identity = digest(captured)
    common = dict(volumes={root:vol},secrets=[modal.Secret.from_name('wandb-secret')],
                  cpu=(4,4),memory=(16384,16384),retries=0,min_containers=0,buffer_containers=0,
                  scaledown_window=2,single_use_containers=True,serialized=True,include_source=False)
    def enter(approved):
        import sys
        os.chdir(code)
        if code not in sys.path:
            sys.path.insert(0,code)
        from rl.campaign import validate_authorization,validate_training_jobs
        validate_authorization(approved)
        validate_training_jobs(approved)
        verify_identity(approved,captured,code)
        vol.reload()
    def claim(approved,name):
        claims=modal.Dict.from_name(approved['deployment_name']+'-owners',create_if_missing=True)
        if not claims.put(name,{'call_id':modal.current_function_call_id(),'identity':identity},skip_if_exists=True):
            raise ValueError('duplicate ownership; automatic retries prohibited')

    @app.function(name='acceptance',gpu='L40S',timeout=600,startup_timeout=600,max_containers=1,**common)
    def acceptance(approved,dispatch_time):
        enter(approved)
        claim(approved,'acceptance')
        import subprocess,sys
        from rl.campaign import hard_deadline
        from rl.checkpointing import atomic_json
        stop=min(dispatch_time+570,approved['deadline_utc_epoch']-30)
        out=Path(root)/'campaigns'/approved['campaign_id']/'acceptance'
        out.mkdir(parents=True,exist_ok=True)
        with hard_deadline(stop):
            env={**os.environ,'RUN_GPU':'1','PYTHONPATH':os.pathsep.join(dict.fromkeys([code,*sys.path]))}
            command=[sys.executable,'-m','pytest','-q','-s','tests/test_gpu_vllm.py','--basetemp',str(out/'pytest')]
            prior=approved.get('verified_adapter_test')
            if prior:
                # Reuse only immutable, hash-verified passing output when every
                # model/trainer/engine/test input is byte-identical. Launcher
                # subprocess-path repairs do not change those tested inputs.
                from rl.checkpointing import file_hash
                previous=Path(root)/'campaigns'/prior['campaign_id']/'acceptance'
                if file_hash(previous/'vllm-tests.txt')!=prior['log_sha256']:
                    raise ValueError('prior adapter proof checksum mismatch')
                previous_manifest=json.loads((previous/'smoke-input.json').read_text())['manifest']
                for name in approved['source_hashes']:
                    if name.startswith('rl/') or name in ('train_grpo.py','tests/test_gpu_vllm.py'):
                        if approved['source_hashes'][name]!=previous_manifest['source_hashes'][name]:
                            raise ValueError('changed tested source requires adapter test rerun')
                if approved['dataset_hashes']!=previous_manifest['dataset_hashes']:
                    raise ValueError('changed tested data requires adapter test rerun')
                text=(previous/'vllm-tests.txt').read_text()
                if '2 passed,' not in text or 'FAILED tests/' in text:
                    raise ValueError('prior adapter tests not passed')
                atomic_json(out/'reused-adapter-proof.json',prior)
            else:
                with (out/'vllm-tests.txt').open('w') as log:
                    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,stop-time.time()-90))
                vol.commit()
                if result.returncode:
                    raise RuntimeError('vLLM acceptance failed; see '+str(out/'vllm-tests.txt'))
            if approved.get('verified_smoke'):
                vol.commit()
                result=verify_saved_smoke(approved,vol,root)
            else:
                job={**approved['jobs'][0],'stop_at':stop}
                payload=out/'smoke-input.json'
                atomic_json(payload,{'manifest':approved,'job':job,'root':root})
                with (out/'smoke-log.txt').open('w') as log:
                    result=subprocess.run([sys.executable,'-m','scripts.deadline_worker','--smoke',str(payload)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,stop-time.time()))
                vol.commit()
                if result.returncode:
                    raise RuntimeError('smoke acceptance failed; see '+str(out/'smoke-log.txt'))
                result=json.loads((out/approved['jobs'][0]['logical_run']/'terminal.json').read_text())
            result['acceptance_seconds']=time.time()-dispatch_time
            result['identity']=identity
            atomic_json(out/'PASSED.json',result)
            vol.commit()
            return result

    @app.function(name='campaign_worker',gpu='L40S',timeout=3600,startup_timeout=300,max_containers=6,**common)
    def worker(job,approved):
        enter(approved)
        if job not in approved['jobs']:
            raise ValueError('unapproved worker configuration')
        claim(approved,job['logical_run'])
        return run_training(approved,job,vol,root)

    @app.function(name='campaign_controller',cpu=(2,2),memory=(8192,8192),timeout=3600,startup_timeout=120,
                  max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
                  single_use_containers=True,retries=0,serialized=True,include_source=False,volumes={root:vol})
    def controller(approved):
        enter(approved)
        claim(approved,'controller')
        from rl.campaign import reserve_training_batch,hard_deadline
        from rl.checkpointing import atomic_json
        base=Path(root)/'campaigns'/approved['campaign_id']
        accepted=json.loads((base/'acceptance/PASSED.json').read_text())
        if accepted.get('identity')!=identity:
            raise ValueError('acceptance identity mismatch')
        projection=max(3300,accepted['max_update_seconds']*200*1.25+900)
        if not approved.get('allow_partial') and time.time()+projection>approved['deadline_utc_epoch']-180:
            raise ValueError('conservative six-run completion projection exceeds cutoff')
        def persist(state):
            atomic_json(base/'controller.json',state)
            vol.commit()
        state=reserve_training_batch(approved,persist)
        calls={}
        with hard_deadline(approved['deadline_utc_epoch']-15):
            try:
                for job in approved['jobs']:
                    timeout=int(approved['deadline_utc_epoch']-time.time()-30)
                    if timeout<=180:
                        raise TimeoutError('insufficient startup/shutdown window')
                    call=worker.with_options(timeout=timeout).spawn(job,approved)
                    calls[job['logical_run']]=call
                    state['jobs'][job['logical_run']].update(status='running',call_id=call.object_id)
                    persist(state)
                pending=dict(calls)
                while pending and time.time()<approved['deadline_utc_epoch']-45:
                    for name,call in list(pending.items()):
                        try:
                            result=call.get(timeout=0)
                        except TimeoutError:
                            continue
                        except Exception as exc:
                            state['jobs'][name].update(status='failed',error=repr(exc))
                            call.cancel(terminate_containers=True)
                            # A shared numerical/configuration failure cannot safely leave other arms running.
                            if any(term in repr(exc).lower() for term in ('nonfinite','fingerprint','configuration','sampler gap')):
                                raise
                        else:
                            state['jobs'][name].update(status='done' if result['step']==200 else 'partial',result=result)
                        pending.pop(name)
                        persist(state)
                    time.sleep(2)
            finally:
                for name,call in calls.items():
                    if state['jobs'][name]['status']=='running':
                        call.cancel(terminate_containers=True)
                        state['jobs'][name]['status']='cancelled-at-cutoff'
                persist(state)
        return state


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',required=True)
    args=parser.parse_args()
    data=json.loads(Path(args.smoke).read_text())
    import modal
    volume=modal.Volume.from_name('lm-rl')
    print(json.dumps(run_training(data['manifest'],data['job'],volume,data['root'],smoke=True)))
