"""Initial closed-owner preparation. All reference graphs/weights remain owned and unchanged."""
import os,hashlib,json,contextlib
from pathlib import Path
from dataclasses import dataclass
from .abc_contract import PROFILES,Selection,descriptor,family_key,lane_rows,digest,capture_policy
from . import abc_runtime as R

@dataclass(frozen=True)
class Plan:
    b:int
    k:int
    profile:object
    @property
    def served_by(self):return descriptor(self.served_b,self.k)
    @property
    def served_b(self):return 8 if self.b==7 and self.k==5 else self.b
    @property
    def components(self):return ('attention','experts','collective')
    @property
    def experts(self):
        # The incumbent already uses its older TC v2 at 48 rows. Dense/wo_a
        # ablations must retain that route; coop has only 256 routing slots.
        return 'tc' if lane_rows(self.profile.experts,self.b,self.k) or (self.profile.experts=='ours' and self.k==5 and self.served_b==8) else 'coop'
    @property
    def collective(self):
        n=(self.k+1)*self.b
        return 'pair' if n==48 else (1,512 if n==6 else 1024,1)

def attest():
    d=json.loads(Path('/opt/star-abc/manifest.json').read_text())
    for row in d['files']:
        if hashlib.sha256(Path(row['target']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('ABC source/binary identity differs: '+row['target'])
    from .abc_kv_contract import requested_dtype
    return digest(dict(source=d['identity'],kv_cache_dtype=requested_dtype()))

def prepare(worker):
    if os.environ.get('STAR_ABC','0')!='1':return dict(enabled=False)
    import gc,torch,torch.distributed as dist
    from vllm.distributed import get_tp_group,graph_capture
    from w9_exl_ab import runtime as original
    from . import abc_kernels as K,abc_gate as G,abc_sampler,r2_bank,r2_experts
    from .w26_dual_graph import Runtime
    from .r2_attention import AttentionStateRows,RowsProfile
    from .initial_state_plan import collect,InitialJournal
    from .attention_qualification import qualify_replay
    from .attention_startup import UnsafePreparation
    from .initial_decode_contract import initial_decode
    if getattr(worker,'_abc_attempted',False):raise RuntimeError('Repeated ABC preparation')
    worker._abc_attempted=True
    runner=worker.model_runner;p=runner.drafter;tp=get_tp_group()
    def consensus(v):
        rows=[None]*tp.world_size;dist.all_gather_object(rows,v,group=tp.cpu_group);return rows
    def agree(ok,phase):
        votes=consensus(dict(ok=bool(ok),phase=phase))
        if len(votes)!=4 or not all(v==votes[0] and v['ok'] for v in votes):raise UnsafePreparation('ABC all-rank phase rejected: '+str(votes))
    roots=worker._w26_cumulative_roots
    def local(phase,fn):
        value=None;error=None
        try:value=fn();roots.append(value)
        except Exception as exc:
            torch.cuda.synchronize();error=repr(exc)
        agree(error is None,phase+': '+str(error))
        return value
    identity=local('ABC attestation',attest)
    agree(len(set(consensus(identity)))==1,'ABC common source identity')
    adapter=worker._w26_attention_adapter
    agree(not getattr(runner,'_w26_serving_seen',True) and adapter.controller.availability().get('next-attention-sampler-native-mixed',False),'ABC requires qualified optimized A before first request')
    r2=getattr(adapter,'r2',None)
    agree(r2 is not None and all(descriptor(8 if b==7 else b) in r2.entries for b in range(2,9)),'ABC requires all existing multiuser reference families')
    bank=R.Bank(worker,adapter.controller.entries['next_mixed'],adapter.controller.sampler_state,r2)
    roots.append(bank);worker._abc_bank=bank
    from .abc_kv_contract import requested_dtype
    scope=os.environ.get('STAR_ABC_SCOPE','all')
    policy=local('ABC declared capture policy',lambda:capture_policy(scope,p.model.model.confidence_head is not None))
    agree(len({digest(x) for x in consensus(policy)})==1,'ABC common declared capture policy')
    batches=policy['batches'];widths=policy['widths']
    receipt=dict(capture_scope=scope,capture_policy=policy,candidate_batch_sizes=batches,candidate_widths=widths,kv_cache_dtype=requested_dtype(),kv_qualification=getattr(runner,'_abc_bf16_kv_receipt',None),enabled=True,ready=False,identity=identity,model_quality_verified=False,profiles={},families={})
    worker._abc_receipt=receipt
    native=adapter.runtime.native
    local('ABC reference binding',adapter.current_identity)
    torch.cuda.synchronize();before_cleanup=torch.cuda.mem_get_info(runner.device)[0]
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    before,total=torch.cuda.mem_get_info(runner.device)
    cap=int(os.environ.get('STAR_ABC_CAPTURE_MIB','2048'))*1024**2
    receipt['initial_memory']=dict(free_before_cleanup=before_cleanup,free=before,total=total,allocated=torch.cuda.memory_allocated(runner.device),reserved=torch.cuda.memory_reserved(runner.device),capture_cap=cap,retained_floor=1024**3)
    print('STAR_ABC_MEMORY '+json.dumps(dict(rank=int(worker.rank),**receipt['initial_memory'])),flush=True)
    agree(before>=cap+1024**3,'ABC capture reserve and retained 1GiB floor')
    gate_receipt=local('ABC router parity',lambda:G.prepare(runner.device))
    agree(len({json.dumps(x,sort_keys=True) for x in consensus(gate_receipt)})==1,'ABC router gate common')
    receipt['woa']=local('ABC exact wo_a recovery',lambda:K.prepare_woa(runner.get_model()))
    receipt['lane']=local('ABC owned lane workspace',lambda:K.prepare_lane(runner.device))
    if os.environ.get('STAR_ABC_OPERATOR_CHECKS','1')=='1':
        from . import abc_lane_check
        receipt['lane_operator_checks']=local('ABC lane reference fixtures',lambda:abc_lane_check.prepare(runner.get_model()))
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    receipt['dense_pack_bytes']=K.PACK_BYTES
    from . import abc_numeric
    receipt['dense_operator_checks']=abc_numeric.DENSE
    journal=local('ABC max geometry state journal',lambda:InitialJournal(collect(runner,phase='initial_before_first_request',rows=48,reqs=8),budget_bytes=128*1024**2))
    def restore():
        journal.restore();G.reset();agree(journal.verify(),'ABC semantic state restoration')
    pool=local('ABC shared private serialized pool',torch.cuda.graph_pool_handle)
    from .abc_attention import Profile as AttnProfile,State as AttnState
    class Context(r2_bank.CaptureContext):
        def _planned_size(self,x):return str(x.dtype)=='torch.bfloat16' and x.numel()*x.element_size()==self.plan.served_by[0]*10240
        def collective_receipt(self):return dict(self.collective,profile=self.plan.collective,bytes=self.plan.served_by[0]*10240,ok=self.collective['planned']>0 and self.collective['unavailable']==0)
    class ABCGraph(Runtime):
        def __init__(self,plan,attention):
            super().__init__(runner,None,native,before)
            self.pool=pool;self.plan=plan;self.profile=plan.profile;self.rows=plan.served_by[0];self.capture_desc=plan.served_by
            self.reference=bank.reference_entry(plan.b,plan.k);self.attention_state=attention;self.paths={}
            self.initial_consensus=consensus;self.initial_metadata_records=[]
        def guard(self,reserve=0):
            torch.cuda.synchronize();free=torch.cuda.mem_get_info(runner.device)[0]
            if before-free+reserve>cap or free-reserve<1024**3:raise ValueError('ABC total retained allocation/floor exceeded')
            if self.baseline_entries!={k:id(v) for k,v in self.wrapper.entries.items()}:raise ValueError('Incumbent entries changed')
            return dict(free=free,growth=max(0,before-free),cap=cap)
        def dummy(self,mode,*,target_only=False):
            from vllm.config import CUDAGraphMode
            with initial_decode(runner,consensus,self.initial_metadata_records,b=self.plan.served_b,query_len=self.plan.k+1):
                return runner._dummy_run(self.rows,cudagraph_runtime_mode=getattr(CUDAGraphMode,mode),force_attention=True,uniform_decode=True,uniform_query_len=self.plan.k+1,allow_microbatching=False,skip_eplb=True,remove_lora=False,num_active_loras=0,is_graph_capturing=mode=='FULL',w26_target_bank_only=True)
        @contextlib.contextmanager
        def candidate_math(self):
            old_bank,old_exp=r2_bank.ACTIVE,r2_experts.ACTIVE
            ctx=Context(self.plan);r2_bank.ACTIVE=ctx;r2_experts.ACTIVE=ctx
            self.capture_context=ctx
            try:
                with super().candidate_math():yield
            finally:r2_bank.ACTIVE=old_bank;r2_experts.ACTIVE=old_exp
    families={p.family():p for p in (PROFILES[x] for x in policy['families'])}
    gate_ok={} if scope!='all' else {family:True for family in families if family in (PROFILES['B'].family(),PROFILES['B-LANE'].family())}
    def gate_probe(rt):
        from .attention_qualification import clone
        from .abc_gate_check import prefix_exact
        results=[]
        for seed in (17,23):
            reference=None
            for m in (5,5,0,1,2,3,4):
                restore()
                runner.input_ids.gpu[:rt.rows].fill_(seed)
                gen=torch.Generator(device=runner.device).manual_seed(seed)
                runner.inputs_embeds.gpu[:rt.rows].copy_(torch.randn(runner.inputs_embeds.gpu[:rt.rows].shape,device=runner.device,dtype=runner.inputs_embeds.gpu.dtype,generator=gen)*.125)
                keep=(torch.arange(rt.rows,device=runner.device)%6<=m)
                G.MASK[:rt.rows].copy_(keep.to(torch.int32))
                rt.mode='qualify_replay';out=clone(rt.dummy('FULL',target_only=True));torch.cuda.synchronize()
                native.check_communication_health()
                if reference is None:reference=out
                exact=prefix_exact(reference,out,keep)
                votes=consensus(bool(exact));results.append(dict(seed=seed,kept_drafts=m,prefix_bitwise=all(votes)))
        rt.mode='baseline';restore()
        return dict(exact_prefix_all=all(r['prefix_bitwise'] for r in results),records=results,model_quality_verified=False)
    # Retain narrow family only if its whole constructor/metadata/replay checks succeed.
    for k in widths:
        for b in reversed(batches):
            for family,profile in families.items():
                if k==3 and family!=PROFILES['B'].family():continue  # narrow/auto only use the combined target family
                plan=Plan(b,k,profile);key=family_key(family,b,k)
                if b==7 and k==5:
                    bank.targets[key]=bank.targets[family_key(family,8,5)]
                    receipt['families'][str(key)]=dict(ready=True,alias_of=str(family_key(family,8,5)),padded_rows=48)
                    continue
                split=(8,8,False) if plan.served_by[0]<=12 else (4,8,False)
                attn=local('ABC attention allocation '+str(key),lambda:AttnState(runner.device,AttnProfile(plan.served_by[0],*split)))
                local('ABC attention compile '+str(key),attn.compile_probe)
                rt=ABCGraph(plan,attn);roots.append(rt);R.CAPTURE=rt
                try:
                    restore();rt.mode='warm'
                    saved_state=rt.attention_state;rt.attention_state=None
                    with rt.candidate_math():rt.dummy('FULL',target_only=True)
                    rt.attention_state=saved_state;torch.cuda.synchronize();restore()
                    rt.paths={};rt.mode='capture'
                    with graph_capture(runner.device,original.S.startup_capture_context):rt.dummy('FULL',target_only=True)
                    agree(True,'ABC capture completed '+str(key));rt.mode='baseline';restore()
                    if b==1 and k==5 and plan.experts!='tc':
                        # Reapply the exact old mixed executable to the new private graph only.
                        from .native_mixed_private import PrivateMixed,artifact
                        from w9_exl_ab import params
                        binary=local('ABC B1 mixed artifact',lambda:artifact(os.environ['W9_VERSION_BUNDLE'],native.R.__file__,params.__file__))
                        mixed=local('ABC B1 mixed module',lambda:PrivateMixed(native).load(binary))
                        local('ABC B1 mixed inventory',lambda:mixed.inventory(rt.entry));local('ABC B1 mixed apply',mixed.apply)
                        rt.entry['abc_mixed']=mixed
                    paths=rt.paths.copy();capture=rt.capture_context
                    if not paths.get('dense_native_fp8') and profile.dense:raise UnsafePreparation('ABC dense candidate captured no native FP8 calls')
                    if not paths.get('woa_native_fp8') and profile.woa and rt.rows<=16:raise UnsafePreparation('ABC wo_a candidate captured no fused calls')
                    if lane_rows(profile.experts,b,k) and not paths.get('experts_lane'):raise UnsafePreparation('ABC lane candidate fell through')
                    comm=capture.collective_receipt()
                    expert_receipt=r2_experts.receipt(capture.experts,plan)
                    agree(expert_receipt['ok'],'ABC routed-expert coverage '+str(key))
                    agree(comm['ok'],'ABC collective coverage '+str(key))
                    agree(len({digest(x) for x in consensus(comm)})==1,'ABC common collective protocol '+str(key))
                    quality=qualify_replay(rt,journal,native,consensus)
                    gating=None
                    if k==5 and family in gate_ok:
                        gating=gate_probe(rt);gate_ok[family]=gate_ok[family] and gating['exact_prefix_all']
                    rt.guard();bank.targets[key]=rt.entry
                    receipt['families'][str(key)]=dict(ready=True,paths=paths,collective=comm,experts=expert_receipt,hidden_replay=quality,gate_prefix=gating)
                except BaseException as exc:
                    # No unilateral continue after uncertain CUDA/collective state.
                    raise UnsafePreparation('ABC family preparation failed '+str(key)) from exc
                finally:rt.mode='baseline';R.CAPTURE=None
    restore()
    from .w26_sampler_bank import eligibility
    _,(start,end)=eligibility(p)
    nccl=getattr(tp.device_communicator,'pynccl_comm',None)
    q8=local('ABC separate draft-only FP8 head',lambda:abc_sampler.quantize_head(p))
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    for key in policy['reference_samplers']:
        state=local('ABC retained original sampler '+str(key),lambda:R.reference_sampler(bank,key[0]))
        bank.samplers[key]=state
        receipt['profiles'][str(key)]=dict(ready=True,reused_qualified_reference=True,fp8_draft=False,fused=False,gate=False,
            qualification='Original B1/R2 sampler proof and immutable binding retained; no new sampler capture')
    for key in policy['samplers']:
        b,fp8,fused,gate=key
        # This phase contains all-gathers; errors are never locally swallowed.
        state,srec=abc_sampler.prepare(p,nccl,start,end,b,fp8,fused,gate,q8,consensus)
        roots.append(state);bank.samplers[key]=state;receipt['profiles'][str(key)]=srec
        agree(True,'ABC sampler completed '+str(key))
    # Fill the existing B1 sampled-capture hole with exactly the established BF16 q/RNG chain.
    from types import SimpleNamespace
    from . import r2_sampler
    def collective(name,fn):return fn()
    sampled_contract=SimpleNamespace(plans=[SimpleNamespace(b=1,components=('sampled',))])
    _,sampled,sampled_receipt=r2_sampler.prepare_rows(p,nccl,sampled_contract,consensus,agree,local,collective,start,end,129280)
    bank.sampled=sampled;receipt['sampled_b1']=sampled_receipt
    agree(1 in sampled,'ABC B1 sampled capture q/RNG parity')
    restore()
    local('ABC untouched original identity',adapter.current_identity)
    available=['A']
    for name,pr in PROFILES.items():
        if name=='A':continue
        if pr.reference:available.append(name);continue
        if pr.gate and not gate_ok.get(pr.family(),False):continue
        if pr.experts=='routed':
            if all(family_key(f,b,5) in bank.targets for f in (PROFILES['B'].family(),PROFILES['B-LANE'].family()) for b in batches):available.append(name)
            continue
        widths=(3,5) if pr.width=='auto' else (pr.width,)
        if all(family_key(pr.family(),b,k) in bank.targets and (b,pr.fp8_draft and b<=3,pr.fused_markov,pr.gate) in bank.samplers for b in batches for k in widths):available.append(name)
    common=dict(source=identity,capture_scope=scope,candidate_batch_sizes=batches,available=available,reference=adapter.current_identity().digest())
    common['rank_receipts']=consensus(digest(receipt))
    bank.selection=Selection(digest(common),available);bank.receipts=receipt
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    receipt.update(ready=True,available=available,identity=bank.selection.identity,memory=rt.guard(),reference_retained=True,default_profile='A',armed=False)
    out=Path('/r1out')/('abc-preparation-rank'+str(worker.rank)+'.json')
    local('ABC durable startup receipt',lambda:out.write_text(json.dumps(receipt,indent=2)))
    R.BANK=bank
    from .prefill_startup import prepare as prefill_prepare  # STAR_PREFILL default-off
    receipt['prefill_preparation']=prefill_prepare(worker)
    receipt['memory']=rt.guard()
    local('ABC final prefill preparation receipt',lambda:out.write_text(json.dumps(receipt,indent=2)))
    return dict(rank=int(worker.rank),ok=True,ready=True,receipt=receipt,identity=bank.selection.identity)

