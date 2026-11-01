"""CPU-testable all-rank two-phase selection protocol for future graph banks.

Caller is a new private control operation, not an existing hot API. Consensus
and drain are supplied by the actual controller. No lazy capture is permitted.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Bundle:
    native_arm:str
    native_version:str
    native_hash:str
    collective_profile:str
    baseline_capture_generation:int
    sampler_version:str
    descriptor:tuple=(6,1,True,False,0)
    attention_version:str='stock'
    mhc_version:str='stock'

class Bank:
    def __init__(self):
        self.entries={};self.selected=None;self.pending=None;self.status='disabled';self.receipt={}
    def disabled(self,reason):
        self.selected=None;self.pending=None;self.status='disabled'
        self.receipt=dict(ready=False,candidate_disabled=True,incumbent_retained=True,reason=str(reason))
    def publish(self,bundle,entry,*,parity,native_patch_qualified,all_rank_ready):
        if not (parity and native_patch_qualified and all_rank_ready):
            self.disabled('candidate qualification incomplete');return False
        if entry is None:raise ValueError('No built graph')
        self.entries[bundle]=entry;self.status='ready';self.receipt=dict(ready=True,selected=False)
        return True
    def prepare_select(self,token,bundle,*,paused,drained,current_bundle):
        if not paused or not drained:raise RuntimeError('Selection requires native pause and completed consumer drain')
        if self.pending is not None:raise RuntimeError('Selection already pending')
        if bundle is not None and (bundle not in self.entries or bundle!=current_bundle):raise ValueError('Missing/stale qualified bundle')
        self.pending=(token,bundle);return True
    def commit_select(self,token,*,all_rank_prepared):
        if not all_rank_prepared or self.pending is None or self.pending[0]!=token:
            self.selected=None;self.pending=None;return False
        self.selected=self.pending[1];self.pending=None;return True
    def rollback_select(self):
        self.selected=None;self.pending=None
    def lookup(self,*,runtime_mode,descriptor,current_bundle):
        if self.pending is not None:raise RuntimeError('Serving forbidden during selection transaction')
        if runtime_mode!='FULL' or descriptor!=(6,1,True,False,0):return None
        if self.selected is None:return None
        if current_bundle!=self.selected:raise RuntimeError('Native/profile bundle changed while candidate selected; controller must invalidate all ranks before mutation')
        return self.entries.get(self.selected)
