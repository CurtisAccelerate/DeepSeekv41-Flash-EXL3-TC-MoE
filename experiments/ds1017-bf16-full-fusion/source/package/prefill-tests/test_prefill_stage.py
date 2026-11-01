import asyncio
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"overlay/usr/local/lib/python3.12/dist-packages"))
from hybrid.prefill_contract import Selection,REFERENCE,MOE_TRY,eligible_geometry
from hybrid.prefill_runtime import Bank,NativeAdapter,binding
from hybrid.prefill_control import Owner


class Tensor:
    serial=100
    def __init__(self,shape=(65,5120),dtype="bf16",device="cuda:0",contiguous=True):
        self.shape,self.dtype,self.device,self.is_cuda,self.contig=shape,dtype,device,True,contiguous
        Tensor.serial+=1;self.pointer=Tensor.serial
    def data_ptr(self):return self.pointer
    def numel(self):
        n=1
        for x in self.shape:n*=x
        return n
    def dim(self):return len(self.shape)
    def is_contiguous(self):return self.contig
    def contiguous(self):return self
    def float(self):return Tensor(self.shape,"fp32",self.device)


class Torch:
    bfloat16="bf16";float32="fp32"
    def __init__(self):
        self.capture=False;self.stream=7
        self.cuda=SimpleNamespace(is_current_stream_capturing=lambda:self.capture,
                                  current_stream=lambda device:SimpleNamespace(cuda_stream=self.stream))
    def empty_like(self,x):return Tensor(x.shape,x.dtype,x.device)


def fixture(rows=65):
    torch=Torch();calls=[];original_calls=[]
    ops=SimpleNamespace(moe_out_prefill_try=lambda *args:calls.append(args))
    layer=SimpleNamespace(_exl3_tc=dict(ptrs=Tensor((384,3,3),"i64"),meta=Tensor((384,3,4),"i64"),
        emap=Tensor((384,),"i32"),workspace=Tensor((150000000,),"fp32"),
        intermediate=640,bits_x2=6,prefill=True,prefill_rows=2048))
    worker=SimpleNamespace(rank=0,_abc_bank=SimpleNamespace(selection=SimpleNamespace(session="bench")))
    original=lambda *args:original_calls.append(args) or "incumbent"
    mode=["NONE"]
    bank=Bank(worker,original,torch,NativeAdapter(torch,ops),{id(layer):binding(layer._exl3_tc)},7,
              lambda:mode[0],"identity",{})
    q=dict(token="tx",profile=MOE_TRY,session="bench",identity="identity",epoch=1)
    bank.selection.prepare(q,"bench");bank.selection.commit("tx")
    args=(Tensor((rows,5120)),Tensor((rows,6),"i64"),Tensor((rows,6),"fp32"),layer,None,7.0)
    return bank,args,calls,original_calls,mode


class Contracts(unittest.TestCase):
    def test_stale_wrong_session_and_unprepared_cannot_select(self):
        selection=Selection("id",False)
        q=dict(token="t",profile=MOE_TRY,session="bench",identity="id",epoch=1)
        with self.assertRaises(ValueError):selection.prepare(q,"bench")
        q["profile"]=REFERENCE
        with self.assertRaises(ValueError):selection.prepare(q,"different")
        q["epoch"]=True
        with self.assertRaises(ValueError):selection.prepare(q,"bench")
        self.assertIsNone(selection.pending)

    def test_runtime_transaction_and_session_ownership(self):
        s=Selection("id",True)
        s.prepare(dict(token="t",profile=MOE_TRY,session="b",identity="id",epoch=1),"b")
        self.assertIsNone(s.record("b")["effective"])
        with self.assertRaises(RuntimeError):s.effective("b")
        with self.assertRaises(ValueError):s.commit("wrong")
        s.commit("t")
        self.assertEqual(s.effective("b"),MOE_TRY)
        self.assertEqual(s.effective(None),REFERENCE)
        self.assertEqual(s.effective("new_session"),REFERENCE)

    def test_default_off_never_imports_torch_or_vllm(self):
        from hybrid import prefill_startup
        with patch.dict(os.environ,{"STAR_PREFILL":"0"}),patch("builtins.__import__",side_effect=AssertionError("imported dependency")):
            self.assertEqual(prefill_startup.prepare(SimpleNamespace())["enabled"],False)


class Dispatch(unittest.TestCase):
    def test_native_abi_preserves_out_workspace_limit_positions(self):
        b,a,calls,old,_=fixture();out=b.apply(*a)
        self.assertEqual(len(calls),1);self.assertEqual(old,[])
        call=calls[0];tc=a[3]._exl3_tc
        self.assertIs(call[8],out);self.assertIs(call[9],tc["workspace"]);self.assertEqual(call[10],7.)
        self.assertIs(call[3],tc["ptrs"]);self.assertIs(call[4],tc["meta"])

    def test_all_decode_rows_and_out_of_range_keep_incumbent(self):
        for rows in (1,6,12,18,24,30,36,42,48,64,2049):
            b,a,calls,old,_=fixture(rows)
            self.assertEqual(b.apply(*a),"incumbent");self.assertFalse(calls);self.assertEqual(len(old),1)
        for rows in (65,2048):
            b,a,calls,old,_=fixture(rows);b.apply(*a)
            self.assertEqual(len(calls),1);self.assertFalse(old)

    def test_capture_graph_and_changed_stream_use_original(self):
        for variant in ("capture","mode","stream"):
            b,a,calls,old,mode=fixture()
            if variant=="capture":b.torch.capture=True
            elif variant=="mode":mode[0]="FULL"
            else:b.torch.stream=9
            self.assertEqual(b.apply(*a),"incumbent");self.assertFalse(calls)

    def test_pointer_movement_is_not_silently_hidden(self):
        b,a,calls,old,_=fixture();a[3]._exl3_tc["workspace"].pointer+=1
        with self.assertRaises(RuntimeError):b.apply(*a)
        self.assertFalse(calls);self.assertFalse(old)

    def test_native_failure_propagates_without_fallback(self):
        b,a,calls,old,_=fixture()
        def failure(*args):raise RuntimeError("native CUDA failure")
        b.adapter.ops.moe_out_prefill_try=failure
        with self.assertRaisesRegex(RuntimeError,"native CUDA failure"):b.apply(*a)
        self.assertFalse(old);self.assertFalse(b.in_call)

    def test_ending_abc_or_new_session_cannot_keep_candidate_enabled(self):
        for session in (None,"new"):
            b,a,calls,old,_=fixture();b.worker._abc_bank.selection.session=session
            self.assertEqual(b.apply(*a),"incumbent");self.assertFalse(calls)

    def test_unprepared_layer_and_k2_k3_geometry(self):
        b,a,calls,old,_=fixture();other=SimpleNamespace(_exl3_tc=a[3]._exl3_tc)
        self.assertEqual(b.apply(*a[:3],other,*a[4:]),"incumbent")
        for i in (512,640):
            for k in (4,6):self.assertTrue(eligible_geometry(1024,5120,6,i,k,384))
        self.assertFalse(eligible_geometry(1024,5120,6,2304,6,96))


class FakeControl:
    def __init__(self,failure=None):
        self.events=[];self.failure=failure
        self.admission=SimpleNamespace(production_active=False,production_transition=False,attention_session="bench")
        self.abc_bank=SimpleNamespace(session="bench",close=self.close)
        self.engine=SimpleNamespace(reset_prefix_cache=self.reset,resume_generation=self.resume,is_paused=self.is_paused)
        self.states=[Selection("identity",True) for _ in range(4)]
    async def close(self):self.events.append("drain_pause_qualify")
    async def reset(self,**kwargs):self.events.append("reset_cache");return self.failure!="cache"
    async def resume(self):self.events.append("resume")
    async def is_paused(self):return False
    async def quarantine(self,reason,poison=False):self.events.append(("quarantine",poison))
    def persist(self):self.events.append("persist")
    def save(self,name,policy):self.events.append("save")
    async def rpc(self,method,args,timeout):
        self.events.append(method)
        rows=[]
        for rank,s in enumerate(self.states):
            if method=="wpf_prepare":s.prepare(args[0],"bench")
            elif method=="wpf_commit":
                if not (self.failure=="commit" and rank==3):s.commit(args[0])
            rows.append(dict(rank=rank,ok=True,ready=True,**s.record("bench")))
        if self.failure=="missing" and method=="wpf_prepare":rows.pop()
        return rows


class Controls(unittest.IsolatedAsyncioTestCase):
    async def test_reset_happens_after_all_rank_commit_before_resume(self):
        c=FakeControl();owner=Owner(c)
        result=await owner.handle_locked(dict(op="prefill_select",session="bench",profile=MOE_TRY))
        self.assertTrue(result["prefix_cache_reset"])
        self.assertLess(c.events.index("drain_pause_qualify"),c.events.index("wpf_prepare"))
        self.assertLess(c.events.index("wpf_commit"),c.events.index("reset_cache"))
        self.assertLess(c.events.index("reset_cache"),c.events.index("resume"))
        self.assertEqual(c.admission.attention_session,"bench")

    async def test_uncertain_outcome_or_failed_cache_stays_quarantined(self):
        for failure in ("missing","commit","cache"):
            c=FakeControl(failure);owner=Owner(c)
            with self.assertRaises(RuntimeError):
                await owner.handle_locked(dict(op="prefill_select",session="bench",profile=MOE_TRY))
            self.assertNotIn("resume",c.events)
            self.assertIn(("quarantine",True),c.events)

    async def test_invalid_envelopes_and_production_do_not_pause_or_mutate(self):
        for q in (dict(op="prefill_select",session="wrong",profile=MOE_TRY),
                  dict(op="prefill_select",session="bench",profile="arbitrary")):
            c=FakeControl();owner=Owner(c)
            with self.assertRaises(ValueError):await owner.handle_locked(q)
            self.assertEqual(c.events,[])
        c=FakeControl();c.admission.production_active=True
        with self.assertRaises(ValueError):
            await Owner(c).handle_locked(dict(op="prefill_select",session="bench",profile=MOE_TRY))
        self.assertEqual(c.events,[])


if __name__=="__main__":unittest.main()
