import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.abc_control import Owner
from hybrid.abc_contract import Selection

class Engine:
    def __init__(self):self.paused=True;self.resumes=0;self.resets=0
    async def resume_generation(self):self.paused=False;self.resumes+=1
    async def is_paused(self):return self.paused
    async def reset_prefix_cache(self,**kw):self.resets+=1;return True
class Control:
    def __init__(self,root):
        self.root=Path(root);self.engine=Engine();self.poisoned=False;self.fail_commit=False
        self.attention_bank=SimpleNamespace(session=None)
        async def drained():pass
        self.admission=SimpleNamespace(open=False,attention_guarded=False,attention_session=None,production_active=False,production_transition=False,wait_drained=drained)
        self.s=[Selection('pin',['A','B','C3']) for _ in range(4)]
    def save(self,name,value):(self.root/name).write_text(json.dumps(value))
    def persist(self):pass
    async def pause(self):self.engine.paused=True
    async def qualify(self):
        if self.poisoned:raise RuntimeError('Poison')
    async def quarantine(self,reason,poison=False):self.poisoned|=poison;self.engine.paused=True;self.admission.open=False
    async def rpc(self,method,args,timeout):
        out=[]
        for rank,s in enumerate(self.s):
            if method=='wabc_prepare':s.prepare(**args[0])
            elif method=='wabc_commit':
                if rank==3 and self.fail_commit:raise RuntimeError('Lost rank after partial commit')
                s.commit(args[0])
            elif method=='wabc_end':s.end()
            out.append(dict(rank=rank,ok=True,**s.record()))
        return out

class OwnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.c=Control(self.tmp.name);self.owner=Owner(self.c)
    async def asyncTearDown(self):self.tmp.cleanup()
    async def begin(self):return await self.owner.handle_locked(dict(op='abc_begin',session='test'))
    async def test_each_selection_resets_prefix_without_reload(self):
        await self.begin();r=await self.owner.handle_locked(dict(op='abc_select',session='test',profile='B',costs=None))
        self.assertTrue(r['prefix_cache_reset']);self.assertEqual(self.c.engine.resets,2)
        self.assertEqual(self.c.engine.resumes,2);self.assertFalse(self.c.admission.open)
        self.assertEqual({s.selected for s in self.c.s},{'B'})
    async def test_wrong_session_does_not_pause_or_select(self):
        await self.begin();before=self.c.engine.resumes
        with self.assertRaises(ValueError):await self.owner.handle_locked(dict(op='abc_select',session='wrong',profile='B',costs=None))
        self.assertEqual(self.c.engine.resumes,before);self.assertEqual({s.selected for s in self.c.s},{'A'})
    async def test_partial_commit_stays_paused_poisoned_and_closed(self):
        await self.begin();self.c.fail_commit=True;before=self.c.engine.resumes
        with self.assertRaises(RuntimeError):await self.owner.handle_locked(dict(op='abc_select',session='test',profile='B',costs=None))
        self.assertTrue(self.c.poisoned);self.assertTrue(self.c.engine.paused);self.assertFalse(self.c.admission.open)
        self.assertIsNone(self.c.admission.attention_session);self.assertEqual(self.c.engine.resumes,before)
    async def test_restart_disarms_persisted_scheduler_policy(self):
        await self.begin();Owner(self.c)
        self.assertFalse(json.loads((self.c.root/'abc-policy.json').read_text())['armed'])
    async def test_failed_cache_reset_cannot_resume_a_committed_arm(self):
        await self.begin();before=self.c.engine.resumes
        async def fail_reset(**kw):return False
        self.c.engine.reset_prefix_cache=fail_reset
        with self.assertRaises(RuntimeError):await self.owner.handle_locked(dict(op='abc_select',session='test',profile='B',costs=None))
        self.assertTrue(self.c.poisoned);self.assertTrue(self.c.engine.paused)
        self.assertEqual(self.c.engine.resumes,before);self.assertIsNone(self.c.admission.attention_session)
    async def test_end_keeps_engine_paused_without_public_approval(self):
        await self.begin();await self.owner.handle_locked(dict(op='abc_end',session='test'))
        self.assertTrue(self.c.engine.paused);self.assertFalse(self.c.admission.open)
        self.assertTrue(all(s.session is None for s in self.c.s))

if __name__=='__main__':unittest.main()
