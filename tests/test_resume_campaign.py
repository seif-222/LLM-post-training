import copy
import json
from types import SimpleNamespace
import pytest
from scripts.resume_campaign import claim_owner,run_controller,validate_manifest
from rl.campaign import polarity_configs

class Claims(dict):
    def put(self,key,value,skip_if_exists):
        if key in self:return False
        self[key]=value;return True


def test_only_same_platform_input_reclaims_ownership():
    c=Claims()
    assert claim_owner(c,'full_s0','call1','source1') is False
    assert claim_owner(c,'full_s0','call1','source1') is True
    for call,source in [('call2','source1'),('call1','source2')]:
        with pytest.raises(ValueError):claim_owner(c,'full_s0',call,source)

@pytest.fixture
def manifest():
    return {'phase':'direct_resume','launch':True,'authorization':'direct user launch',
            'additional_budget_usd':30,'reserved_usd':27.166704,'overhead_usd':4.5,
            'worker_max_usd_per_second':0.000839504,'authorized_at_epoch':100,'deadline_epoch':4600,
            'scientific_campaign_id':'old','continuation_id':'new','source_hashes':{'x':'a'},'dataset_hashes':{'gsm8k':'a'},
            'jobs':[dict(logical_run=c['logical_run'],argv=c['argv'],checkpoint_step=50,checkpoint_manifest_sha256='a'*64) for c in polarity_configs('old')]}

def test_unchanged_science_and_budget_required(manifest):
    validate_manifest(manifest,now=101)
    for field,value in [('additional_budget_usd',31),('reserved_usd',1),('deadline_epoch',4601),('launch',False)]:
        bad=copy.deepcopy(manifest);bad[field]=value
        with pytest.raises(ValueError):validate_manifest(bad,now=101)
    bad=copy.deepcopy(manifest);bad['jobs'][0]['argv']+=['--lr','0.002']
    with pytest.raises(ValueError):validate_manifest(bad,now=101)

class Call:
    def __init__(self,name):self.object_id=name;self.ready=False;self.cancelled=False;self.error=None
    def get(self,timeout):
        if self.error:raise self.error
        if not self.ready:raise TimeoutError()
        return {'step':200}
    def cancel(self,**kwargs):self.cancelled=True


def setup_controller(tmp_path):
    path=tmp_path/'controller.json';calls={};spawns=[]
    def persist(state):path.write_text(json.dumps(state))
    def spawn(job):
        name=job['logical_run'];spawns.append(name);calls[name]=Call(name);return calls[name]
    return path,calls,spawns,persist,spawn


def test_controller_interruption_keeps_workers_and_recovers_without_spawn(manifest,tmp_path):
    path,calls,spawns,persist,spawn=setup_controller(tmp_path)
    def interrupt(_):raise KeyboardInterrupt('controller preempted')
    with pytest.raises(KeyboardInterrupt):run_controller(manifest,path,persist,spawn,calls.__getitem__,sleep=interrupt,now=lambda:200)
    assert len(spawns)==6 and not any(c.cancelled for c in calls.values())
    for c in calls.values():c.ready=True
    state=run_controller(manifest,path,persist,spawn,calls.__getitem__,now=lambda:200)
    assert len(spawns)==6 and all(r['status']=='done' for r in state['jobs'].values())


def test_isolated_failure_does_not_cancel_other_workers(manifest,tmp_path):
    path,calls,spawns,persist,spawn=setup_controller(tmp_path)
    def dispatch(j):
        c=spawn(j);c.ready=True
        if len(spawns)==1:c.error=RuntimeError('isolated worker failure')
        return c
    result=run_controller(manifest,path,persist,dispatch,calls.__getitem__,now=lambda:200)
    assert sum(r['status']=='done' for r in result['jobs'].values())==5
    assert sum(c.cancelled for c in calls.values())==1


def test_shared_correctness_failure_cancels_others(manifest,tmp_path):
    path,calls,spawns,persist,spawn=setup_controller(tmp_path)
    def dispatch(j):
        c=spawn(j)
        if len(spawns)==1:c.error=FloatingPointError('nonfinite gradient')
        return c
    with pytest.raises(FloatingPointError):run_controller(manifest,path,persist,dispatch,calls.__getitem__,now=lambda:200)
    assert all(c.cancelled for c in calls.values())


def test_cutoff_is_absolute_and_cancels_pending(manifest,tmp_path):
    path,calls,spawns,persist,spawn=setup_controller(tmp_path)
    result=run_controller(manifest,path,persist,spawn,calls.__getitem__,now=lambda:4590)
    assert all(c.cancelled for c in calls.values())
    assert all(r['status']=='cancelled-deadline' for r in result['jobs'].values())


def test_uncertain_dispatch_never_restarts(manifest,tmp_path):
    path,calls,spawns,persist,spawn=setup_controller(tmp_path)
    persist({'continuation_id':'new','jobs':{j['logical_run']:{'status':'dispatching','call_id':None} for j in manifest['jobs']}})
    with pytest.raises(RuntimeError,match='uncertain'):run_controller(manifest,path,persist,spawn,calls.__getitem__)
    assert not spawns
