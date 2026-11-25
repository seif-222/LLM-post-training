import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from rl.campaign import RUNS,polarity_configs,validate_authorization,validate_training_jobs,reserve_training_batch,digest
from scripts.deadline_worker import verify_identity,install

@pytest.fixture
def manifest():
    now=time.time()
    cfg=polarity_configs('deadline-test')
    return dict(phase='training_only',campaign_id='deadline-test',deployment_name='deadline-test',launch=True,
                authorization='test user approval',budget_usd=25,verified_credit_usd=30,deadline_utc_epoch=now+3200,
                credit_expiry_utc_epoch=now+3500,max_gpu_jobs=6,retries=0,runs=list(RUNS),waves=[list(RUNS)],
                jobs=[dict(kind='train',logical_run=c['logical_run'],argv=c['argv'],allocation_usd=3.4) for c in cfg],
                overhead_reservation_usd=4.5,source_hashes={'x':'test'},dataset_hashes={'gsm8k':'test'},allow_partial=True)

def test_six_atomic_and_seventh_rejected(manifest):
    writes=[]
    state=reserve_training_batch(manifest,lambda s:writes.append(copy.deepcopy(s)))
    assert len(writes)==1 and len(state['jobs'])==6 and state['reserved_usd']==pytest.approx(24.9)
    manifest['jobs'].append(copy.deepcopy(manifest['jobs'][0]))
    with pytest.raises(ValueError):reserve_training_batch(manifest,writes.append)
    assert len(writes)==1

@pytest.mark.parametrize('field,value',[('launch',False),('verified_credit_usd',None),('credit_expiry_utc_epoch',0),('deadline_utc_epoch',1),('retries',1),('budget_usd',26)])
def test_authorization_rejection(manifest,field,value):
    manifest[field]=value
    with pytest.raises(ValueError):validate_authorization(manifest)

def test_wrong_config_and_evaluation_rejected(manifest):
    manifest['jobs'][0]['argv']+=['--lr','0.0002']
    with pytest.raises(ValueError):validate_training_jobs(manifest)
    manifest['jobs'][0]['kind']='evaluation'
    with pytest.raises(ValueError):validate_training_jobs(manifest)

def test_identity_mismatch(manifest,tmp_path):
    from rl.checkpointing import file_hash
    (tmp_path/'x').write_text('correct')
    (tmp_path/'manifests').mkdir()
    (tmp_path/'manifests/gsm8k.json').write_text('{}')
    manifest['source_hashes']={'x':file_hash(tmp_path/'x')}
    manifest['dataset_hashes']={'gsm8k':file_hash(tmp_path/'manifests/gsm8k.json')}
    verify_identity(manifest,copy.deepcopy(manifest),tmp_path)
    (tmp_path/'x').write_text('wrong')
    with pytest.raises(ValueError):verify_identity(manifest,copy.deepcopy(manifest),tmp_path)

def test_real_controller_six_dispatch_no_retry(manifest,tmp_path,monkeypatch):
    import modal
    import scripts.deadline_worker as dw
    calls=[]; registry={}; owners=set()
    class Call:
        def __init__(self,n):self.object_id='fake-'+n
        def get(self,timeout):return {'step':200}
        def cancel(self,**kw):pass
    class Function:
        def __init__(self,f,name):self.f=f;self.name=name
        def with_options(self,**kw):return self
        def spawn(self,job,approved):
            calls.append(job);return Call(job['logical_run'])
    class App:
        def function(self,**kw):
            def deco(f):
                registry[kw['name']]=Function(f,kw['name']);return registry[kw['name']]
            return deco
    class Claims:
        def put(self,key,value,skip_if_exists):
            if key in owners:return False
            owners.add(key);return True
    monkeypatch.setattr(modal.Dict,'from_name',lambda *a,**k:Claims())
    monkeypatch.setattr(modal.Secret,'from_name',lambda *a,**k:object())
    monkeypatch.setattr(modal,'current_function_call_id',lambda:'fake-controller')
    monkeypatch.setattr(dw,'verify_identity',lambda *a:None)
    monkeypatch.setattr(dw.time,'sleep',lambda *a:None)
    vol=SimpleNamespace(commit=lambda:None,reload=lambda:None)
    accept=tmp_path/'campaigns'/manifest['campaign_id']/'acceptance'
    accept.mkdir(parents=True)
    (accept/'PASSED.json').write_text(json.dumps(dict(identity=digest(manifest),max_update_seconds=15)))
    old=Path.cwd()
    try:
        install(App(),vol,manifest,str(tmp_path),str(tmp_path))
        result=registry['campaign_controller'].f(manifest)
        assert len(calls)==6 and all(j['kind']=='train' for j in calls)
        assert all(j['status']=='done' for j in result['jobs'].values())
        with pytest.raises(ValueError,match='duplicate ownership'):registry['campaign_controller'].f(manifest)
        assert len(calls)==6
    finally:
        import os
        os.chdir(old)

def test_unauthorized_import_no_construction(monkeypatch):
    import modal,importlib
    monkeypatch.delenv('LM_RL_LAUNCH_MANIFEST',raising=False)
    def forbidden(*a,**kw):raise AssertionError('remote construction')
    monkeypatch.setattr(modal,'App',forbidden)
    import modal_app
    importlib.reload(modal_app)
    assert modal_app.app is None
