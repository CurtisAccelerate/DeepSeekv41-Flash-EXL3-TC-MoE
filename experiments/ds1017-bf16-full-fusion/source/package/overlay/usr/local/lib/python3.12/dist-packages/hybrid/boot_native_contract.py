"""First attention-bank contract: transpose/boot + six, incumbent sampler.

Reads explicit graphs and existing module roots; NEVER calls W9 inventory/apply,
register, begin_capture, or module load/unload. Does not update an executable.
The candidate must freshly capture the same qualified boot functions/profile.
"""
from collections import Counter
import os
from .private_nodes import Driver, PrivateUpdates, nodes, read_values, pointer, Rejected

COUNTS={'rot':40,'a2':5,'a3':35,'b2':5,'b3':35}
DESC=(6,1,True,False,0)

def desc(d):return (d.num_tokens,d.num_reqs,d.uniform,d.has_lora,d.num_active_loras)

class BootNativeContract:
    def __init__(self,runner):
        from w9_exl_ab import runtime as R
        from w9_exl_ab import collective_profiles as CP
        self.R=R;self.CP=CP;self.runner=runner;self.backend=R.S.backend
        self.driver=Driver();self.private=[]
        self.require_current()
        wrapper=runner.model
        matches=[(k,e) for k,e in wrapper.entries.items() if isinstance(k,tuple) and len(k)==3 and k[:2]==('transpose','boot') and desc(k[2])==DESC]
        if len(matches)!=1:raise Rejected('Missing unique target FULL T6 baseline')
        self.key,self.entry=matches[0]
        cap=self.entry.capture
        self.baseline_graphs=tuple(cap.kernel_patchable_graphs)
        if cap.num_graphs!=1 or len(self.baseline_graphs)!=1:raise Rejected('Only retained single FULL graph supported')
        self.baseline_ids={g.raw_cuda_graph() for g in self.baseline_graphs}
        # These are initialized boot modules explicitly retained until CUDA
        # context destruction by the existing Backend, not temporary versions.
        self.roots=(self.backend,tuple(CP.REG),self.entry,self.baseline_graphs)
        self.snapshot=self.state_signature()
        self.desired_collectives=self.collective_signature(self.baseline_graphs,baseline_raw=True)
        if not any(k[1]==61440 for k in self.desired_collectives):raise Rejected('No six-row target collective coverage')
        self.baseline_native=self.native_inventory(self.baseline_graphs)

    def require_current(self):
        from .cumulative_native import assert_sampler_family
        assert_sampler_family(self.runner)
        s=self.R.S;c=self.CP
        if not s.ready or s.poisoned or s.capturing or s.pending or s.version_pending or s.quarantine_reason:
            raise Rejected('Baseline not idle/healthy')
        if (s.arm,s.active_versions.get('transpose'),s.active_hashes.get('transpose'))!=('transpose','boot',self.R.HASHES['transpose']):
            raise Rejected('Only attested transpose boot baseline supported')
        if c.PROFILE!='six' or c.TX is not None or not c.REG:raise Rejected('Expected committed six profile')
        if s.capture_version is not None or s.version_capture is not None:raise Rejected('Version capture is pending')
        if any(os.environ.get(k,'0')!='0' for k in ('W26_DSPARK_LOCAL','W26_DSPARK_CAPTURE_LOCAL','W26_MHC_DUAL_GRAPH','W26_OVERLAP')) or getattr(self.runner.drafter,'_w26_capture_local',None) is not None:
            raise Rejected('Non-incumbent sampler integration requires a separate bundle')

    def state_signature(self):
        s=self.R.S;c=self.CP
        return (s.capture_generation,tuple(sorted((k,tuple(sorted(repr(x) for x in v))) for k,v in s.graphs.items())),
                tuple(sorted((repr(k),id(v)) for k,v in s.graph_refs.items())),
                tuple(sorted((repr(k),id(v)) for k,v in s.all_graph_refs.items())),
                tuple(sorted((repr(k),id(v),id(v.capture)) for k,v in self.runner.model.entries.items())),
                id(c.BANK),repr(c.SIGNATURE),tuple(id(x) for x in c.REG),c.PROFILE,
                tuple(sorted((k,pointer(v)) for k,v in self.backend.functions['transpose'].items())))

    def assert_baseline_unchanged(self):
        self.require_current()
        if self.snapshot!=self.state_signature():raise Rejected('Baseline registrations/entries/profile/module functions changed')

    def check_communication_health(self):
        # Caller synchronizes completed work first. Never restore/rewind tickets.
        from w9_exl_ab import pair_transport as PT
        PT.check_errors(force=True)
        for reg in self.CP.REG:
            owner=reg['owner'];flags=owner.error_flags()
            if owner.disabled or len(flags)!=4 or any(flags):raise RuntimeError('Host collective unhealthy; no optional recovery claim')

    def native_inventory(self,graphs):
        funcs=self.backend.functions['transpose']
        transaction=PrivateUpdates(self.driver,graphs,(self.backend,))
        counts=transaction.native(self.backend.Params,funcs,funcs,COUNTS)
        if any(item[3].func!=item[4].func for item in transaction.items):raise Rejected('Boot function equality failed')
        # No apply call. Deep-owned Params + explicit graph/module roots are
        # retained for lifetime attestation only.
        return transaction

    def collective_signature(self,graphs,*,baseline_raw):
        result=Counter();regs=tuple(self.CP.REG)
        mapping={}
        for index,reg in enumerate(regs):
            for algo,fn in reg['funcs'].items():mapping.setdefault(pointer(fn),[]).append((index,reg,algo))
        for graph in graphs:
            for _,p in nodes(self.driver,graph):
                if p.func not in mapping:continue
                values=read_values(p)
                choices=[(i,r,a) for i,r,a in mapping[p.func] if (r['owner'].dev_base,r['owner'].seq.data_ptr(),r['owner'].rank,r['owner'].slot_bytes)==(values[2],values[3],values[4],values[6])]
                if len(choices)!=1:raise Rejected('Ambiguous host collective owner')
                index,reg,algo=choices[0];size=values[5]*16
                if size<=0 or size>reg['owner'].host_cap or size%16:raise Rejected('Collective extent differs')
                geometry=(p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes)
                if baseline_raw and size==61440:
                    # Raw graph remains stock after executable updates. The
                    # existing committed PROFILE=six is the separate executable
                    # transaction proof; normalize only its exact known effect.
                    if algo!=0 or geometry[:6]!=(1,1,1,1024,1,1):raise Rejected('Unexpected raw baseline six-row geometry')
                    algo=1;geometry=(1,1,1,512,1,1,p.sharedMemBytes)
                if not baseline_raw and size==61440 and (algo!=1 or geometry[:6]!=(1,1,1,512,1,1)):
                    raise Rejected('Candidate did not inherit six profile')
                result[(index,size,algo,geometry)]+=1
        return result

    def qualify_private(self,entry):
        self.assert_baseline_unchanged()
        graph=entry['graph']
        if graph.raw_cuda_graph() in self.baseline_ids:raise Rejected('Refusing baseline graph as candidate')
        native=self.native_inventory((graph,))
        collectives=self.collective_signature((graph,),baseline_raw=False)
        if collectives!=self.desired_collectives:raise Rejected('Candidate collective coverage/profile differs')
        # Strong lifetime roots attached to graph owner; no independent module
        # cleanup is possible while baseline/candidate context remains alive.
        entry['native_roots']=(self.roots,native,tuple(self.CP.REG))
        self.private.append(entry);self.assert_baseline_unchanged()
        return dict(qualified=True,native_counts=COUNTS,collectives=repr(collectives),native_version='boot',collective_profile='six',executable_updates=0,module_loads=0)
