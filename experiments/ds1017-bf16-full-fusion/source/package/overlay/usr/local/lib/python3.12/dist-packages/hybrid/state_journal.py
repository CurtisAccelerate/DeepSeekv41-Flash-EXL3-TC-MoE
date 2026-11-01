"""Staged exact byte journal. No guessing of touched state; caller supplies views.

Not installed or GPU-tested. Non-contiguous views are rejected so each declared
view is one exact byte interval. Alias overlap is merged within the SAME storage.
The complete touched-view census is a separate qualification gate.
"""
def merge_intervals(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not (isinstance(start,int) and isinstance(end,int) and 0<=start<end):raise ValueError('Invalid byte interval')
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((start,end))
    return result

def require_initial_empty_state(*,request_count,request_map_size,cpu_block_ids,device_block_ids,phase,expected_groups=6):
    if phase!='initial_before_first_request':raise ValueError('Current active/prefix state is not startup state')
    if request_count or request_map_size:raise ValueError('Runner still owns requests')
    if len(cpu_block_ids)!=expected_groups or len(device_block_ids)!=expected_groups or any(not row for row in cpu_block_ids+device_block_ids):raise ValueError('Missing all-group dummy block-table proof')
    if any(int(v)!=0 for row in cpu_block_ids for v in row) or any(int(v)!=0 for row in device_block_ids for v in row):raise ValueError('Dummy ring slots could target real blocks')

class ByteJournal:
    def __init__(self,named_views,*,budget_bytes=16*1024**2,free_floor=1024**3,forbidden_device_ranges=()):
        import torch
        self.torch=torch;self.groups={};self.copies=[];self.receipt=[]
        for name,view in named_views:
            if not view.is_cuda or not view.is_contiguous() or view.numel()==0:raise ValueError('Only explicit nonempty contiguous CUDA views supported')
            storage=view.untyped_storage();base=storage.data_ptr();start=view.data_ptr()-base;end=start+view.numel()*view.element_size()
            if end>storage.nbytes():raise ValueError('View exceeds storage')
            for device,lo,hi in forbidden_device_ranges:
                if str(view.device)==str(device) and max(base+start,lo)<min(base+end,hi):raise ValueError('Journal overlaps excluded communication protocol state: '+name)
            group=self.groups.setdefault((str(view.device),base),dict(storage=storage,device=view.device,intervals=[]))
            group['intervals'].append((start,end));self.receipt.append(dict(name=name,storage=base,start=start,end=end))
        if not self.groups:raise ValueError('Missing explicit touched-state census')
        total=sum(end-start for g in self.groups.values() for start,end in merge_intervals(g['intervals']))
        if total>budget_bytes:raise ValueError('Journal exceeds optional memory budget')
        devices={g['device'] for g in self.groups.values()}
        for device in devices:
            if torch.cuda.mem_get_info(device)[0]<free_floor+total:raise ValueError('Journal reserve/floor unavailable')
        for group in self.groups.values():
            for start,end in merge_intervals(group['intervals']):
                view=torch.empty(0,device=group['device'],dtype=torch.uint8).set_(group['storage'],start,(end-start,),(1,))
                self.copies.append((view,view.clone()))
        self.bytes=total
    def restore(self):
        for view,copy in self.copies:view.copy_(copy)
    def verify(self):
        return all(self.torch.equal(view,copy) for view,copy in self.copies)

class BindingJournal:
    """Re-resolve semantic owners and tensor storage, not just saved slices.

    Tensor descriptors contain metadata only. Collection sizes detect added or
    removed traversal entries; per-element getters attest each traversed object.
    Does not restore/rebind tensors or modules after an identity violation.
    """
    @staticmethod
    def signature(value):
        if all(hasattr(value,k) for k in ('data_ptr','shape','stride','storage_offset','untyped_storage')):
            storage=value.untyped_storage()
            return ('tensor',value.data_ptr(),tuple(value.shape),tuple(value.stride()),
                    value.storage_offset(),str(value.dtype),str(value.device),storage.data_ptr(),storage.nbytes())
        if isinstance(value,(list,tuple,dict,set)):return ('container',len(value))
        return ('object',)
    def __init__(self,bindings):
        self.saved=tuple((name,getter,value,self.signature(value)) for name,getter,value in bindings)
    def verify(self):
        try:
            return all((current:=getter()) is value and self.signature(current)==signature
                       for name,getter,value,signature in self.saved)
        except (AttributeError,KeyError,IndexError,RuntimeError,TypeError):return False

class AttributeJournal:
    """Only explicitly named Python aliases; never replace underlying weights."""
    def __init__(self,pairs):self.saved=[(obj,name,hasattr(obj,name),getattr(obj,name,None)) for obj,name in pairs]
    def restore(self):
        for obj,name,existed,value in self.saved:
            if existed:setattr(obj,name,value)
            elif hasattr(obj,name):delattr(obj,name)
    def verify(self):
        return all(hasattr(obj,name)==existed and (not existed or getattr(obj,name) is value) for obj,name,existed,value in self.saved)

if __name__=='__main__':
    import json,pathlib
    assert merge_intervals([(2,8),(0,4),(12,15),(15,16)])==[(0,8),(12,16)]
    require_initial_empty_state(request_count=0,request_map_size=0,cpu_block_ids=[[0]]*6,device_block_ids=[[0]]*6,phase='initial_before_first_request')
    rejected=[]
    for change in [dict(request_count=1),dict(request_map_size=1),dict(cpu_block_ids=[[3]]),dict(device_block_ids=[[2]]),dict(phase='paused_after_requests')]:
        args=dict(request_count=0,request_map_size=0,cpu_block_ids=[[0]]*6,device_block_ids=[[0]]*6,phase='initial_before_first_request');args.update(change)
        try:require_initial_empty_state(**args);raise AssertionError('Unsafe initial guard accepted')
        except ValueError:rejected.append(next(iter(change)))
    result=dict(cpu_pass=True,byte_merge_pass=True,rejected=rejected,gpu_journal_qualified=False,complete_model_write_census=False)
    pathlib.Path(__file__).with_name('state-journal-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
