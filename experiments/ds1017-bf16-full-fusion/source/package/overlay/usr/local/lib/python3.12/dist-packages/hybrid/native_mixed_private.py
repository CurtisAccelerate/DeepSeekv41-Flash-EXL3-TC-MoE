"""Startup-only private executable update. No W9 registration or global selection."""
import ctypes as C
import hashlib,json,os
from pathlib import Path
from .private_nodes import PrivateUpdates,Rejected,pointer
from .boot_native_contract import COUNTS,DESC

PIN='edc623dbeea0931aa37507ed8b2b78f12df054a3756d64507479fcab6c8529e3'
MANIFEST_PIN='38910deb255139ced1775aa768027a68692d92a42d98203c751c972f60d58efd'
PROOF_PIN='e67826fdece6d2d48897bba07aafa25960b92561054050cbbfc54e93fe13098d'
PARAMS_PIN='aa733d2dd34850a0a9c047043ce979250f1a1b3985b551b98dd6247b414a2a83'
RUNTIME_PIN='e77a0c3d0e71bf672ef1cf677cc3b7c0eb6bd5080201bb76cbb8cc39e9131149'
BASELINE_PIN='506f7ccd0108693e8c9f97f866330c976d72fbfca8e093e414f2150962d7560f'
ABI='MoeCoopParams344-SM80-N128-T512-Ksplit1'
VERSION='w25-native-mixed'

def requested(manifest):
    value=manifest.get('next_native_mixed')
    if value is None:return False
    expected=dict(version=VERSION,cubin_sha256=PIN,qualification_sha256=PROOF_PIN)
    if value!=expected:raise ValueError('Unsupported private native profile')
    return True

def artifact(root,runtime_path,params_path):
    root=Path(root).resolve()
    def read(name,pin):
        p=(root/name).resolve()
        if p.parent!=root or hashlib.sha256(p.read_bytes()).hexdigest()!=pin:
            raise ValueError('Private native artifact path/hash differs: '+name)
        return p
    m=json.loads(read(VERSION+'.json',MANIFEST_PIN).read_text())
    binary=read(VERSION+'.cubin',PIN)
    q=json.loads(read(VERSION+'-qualification.json',PROOF_PIN).read_text())
    if (m.get('abi'),m.get('cubin'),m.get('cubin_sha256'),m.get('qualification'),m.get('qualification_sha256'))!=(ABI,binary.name,PIN,VERSION+'-qualification.json',PROOF_PIN):raise ValueError('Private native manifest differs')
    if q.get('cubin_sha256')!=PIN or q.get('primary_bitwise_pass') is not True or q.get('sm80') is not True or q.get('params_bytes')!=344:raise ValueError('Native qualification absent')
    if q.get('params_source_sha256')!=PARAMS_PIN or q.get('baseline_sha256')!=BASELINE_PIN or {tuple(x) for x in q['qualified_bits_widths']}!={(2,512),(2,640),(3,512),(3,640)}:raise ValueError('Native qualification control/shape differs')
    for path,pin in ((runtime_path,RUNTIME_PIN),(params_path,PARAMS_PIN)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=pin:raise ValueError('Loaded native ABI/dispatch source differs')
    return binary

class PrivateMixed:
    """Retained before load; driver errors are unsafe, never optional rejection.

    Raw graph nodes remain boot. Executable state has no CUDA getter: attest the
    completed update ledger and immutable handles, not an invented readback.
    """
    def __init__(self,native):
        self.native=native;self.driver=native.driver;self.module=C.c_void_p()
        self.functions={};self.transaction=None;self.applied=False;self.signature=None
    def load(self,path):
        self.driver.call('cuModuleLoad',C.byref(self.module),C.c_char_p(os.fsencode(path)))
        names={'rot':self.native.R.ROT}
        for k in (2,3):
            for stage in ('a','b'):names[f'{stage}{k}']=f'_ZN16exl3_moe_coop_ns22exl3_moe_coop_{stage}_kernelILi{k}ELi2ELb1ELb0EEEv13MoeCoopParams'
        for name,symbol in names.items():
            fn=C.c_void_p();self.driver.call('cuModuleGetFunction',C.byref(fn),self.module,C.c_char_p(symbol.encode()));self.functions[name]=fn
        return self
    def inventory(self,entry):
        if self.transaction is not None:raise Rejected('Private native update already inventoried')
        boot=self.native.qualify_private(entry)
        tx=PrivateUpdates(self.driver,(entry['graph'],),(self,self.native,self.module))
        self.transaction=tx # retain partial inventory and arguments on any exit
        entry['native_roots']=(entry['native_roots'],self,tx)
        counts=tx.native(self.native.backend.Params,self.native.backend.functions['transpose'],self.functions,COUNTS)
        return dict(boot=boot,native_counts=counts,native_version=VERSION,cubin_sha256=PIN,descriptor=DESC)
    def apply(self):
        n=self.transaction.apply()
        if n!=120:raise RuntimeError('Incomplete private native executable update')
        self.driver.synchronize();self.applied=True
        self.signature=self.binding()
        return dict(executable_updates=n,module_loads=1,raw_graph_native='boot',executable_native=VERSION,readback='completed CUDA update ledger; no executable getter')
    def binding(self):
        tx=self.transaction
        return (pointer(self.module),tuple(sorted((k,pointer(v)) for k,v in self.functions.items())),
            tuple(tx.signature),tx.plan_binding(),tuple(id(x) for x in tx.changed))
    def assert_intact(self):
        self.native.assert_baseline_unchanged()
        if not self.applied or len(self.transaction.changed)!=120 or self.signature!=self.binding():raise RuntimeError('Private native executable binding changed')
        self.transaction.check_handles()
    def rollback(self):
        # Only after completed semantic rejection with healthy CUDA. Unknown
        # driver/apply/communication faults escape without attempted recovery.
        if self.transaction is not None:self.transaction.rollback()
        self.applied=False;self.signature=None

class ReferenceNext:
    """Use the existing next graph as qualify_replay's reference, same journal."""
    def __init__(self,rt,entry):self.rt=rt;self.reference=entry
    def __getattr__(self,k):return getattr(self.rt,k)
    @property
    def mode(self):return self.rt.mode
    @mode.setter
    def mode(self,v):self.rt.mode=v
    def dummy(self,*args,**kwargs):
        saved=self.rt.entry;mode=self.rt.mode
        try:
            if mode=='baseline':self.rt.entry=self.reference;self.rt.mode='qualify_replay'
            return self.rt.dummy(*args,**kwargs)
        finally:self.rt.entry=saved;self.rt.mode=mode

def qualify_next_exact(rt,journal,native,consensus,reference,qualify):
    from .attention_qualification import QualificationRejected
    result=qualify(ReferenceNext(rt,reference),journal,native,consensus)
    rows=result.get('hidden_metrics',[])
    exact=(result.get('benchmark_component_replay_pass') is True and
        [row.get('seed') for row in rows]==[17,23,17] and
        all(row.get('own_replays_exact') is True and row.get('hidden',{}).get('outputs') and
            all(x.get('bit_exact') is True for x in row['hidden']['outputs']) for row in rows))
    votes=consensus(dict(phase='mixed-versus-existing-next-exact',ok=exact))
    if len(votes)!=4 or not all(x.get('phase')=='mixed-versus-existing-next-exact' and x.get('ok') is True for x in votes):raise QualificationRejected('Mixed native differs from existing next capture or proof is incomplete')
    result['mixed_vs_existing_next_bit_exact']=True
    return result
