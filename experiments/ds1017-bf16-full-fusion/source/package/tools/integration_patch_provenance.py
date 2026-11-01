"""Historical V1 asserted integration provenance. Frozen full source is authoritative.
V2 fidelity/probe hooks and numerical fixtures are additional explicit source edits.
"""
from pathlib import Path
import ast,json,hashlib
root=Path(__file__).resolve().parents[1]
pkg=root/'overlay/usr/local/lib/python3.12/dist-packages'

def patch(path,pairs,marker):
    s=path.read_text(encoding='utf-8')
    if marker in s:return
    for old,new in pairs:
        if s.count(old)!=1:raise RuntimeError(f'{path}: expected exactly one {old[:100]!r}, got {s.count(old)}')
        s=s.replace(old,new)
    ast.parse(s,filename=str(path));path.write_text(s,encoding='utf-8')

patch(pkg/'vllm/model_executor/kernels/linear/mxfp8/marlin.py',[
 ('        prepare_mxfp8_layer_for_marlin(layer)','        from hybrid.abc_kernels import prepare_dense  # STAR_ABC integration\n        prepare_dense(layer)\n        prepare_mxfp8_layer_for_marlin(layer)'),
 ('        return apply_mxfp8_marlin_linear(','        from hybrid.abc_kernels import dense\n        candidate = dense(layer, x, bias)\n        if candidate is not None:\n            return candidate\n        return apply_mxfp8_marlin_linear(')
], 'STAR_ABC integration')
patch(pkg/'vllm/models/deepseek_v4_1/ampere/ampere_sparse.py',[
 ('        return super()._o_proj(o, positions)','        from hybrid.abc_kernels import woa  # STAR_ABC integration\n        candidate = woa(self, o, positions)\n        if candidate is not None:\n            return candidate\n        return super()._o_proj(o, positions)')
],'STAR_ABC integration')
patch(pkg/'vllm_exl3/exl3.py',[
 ('    is_mixedk = getattr(layer, "_exl3_mixedk_unified", False)','    from hybrid.abc_kernels import experts  # STAR_ABC integration\n    candidate = experts(x2d, ids, weights, layer, expert_map, limit)\n    if candidate is not None:\n        return candidate\n    is_mixedk = getattr(layer, "_exl3_mixedk_unified", False)')
],'STAR_ABC integration')
patch(pkg/'hybrid/dspark_proposer.py',[
 ('        from hybrid.w26_sampler_bank import try_sample\n        if self._dsv41_r1_plan is not None:', '        from hybrid.abc_runtime import try_sample as abc_sample  # STAR_ABC integration\n        candidate = abc_sample(self, num_reqs, sample_hidden)\n        if candidate is not None:\n            return candidate\n        from hybrid.w26_sampler_bank import try_sample\n        if self._dsv41_r1_plan is not None:'),
 ('        assert num_speculative_tokens == self.num_speculative_tokens, (','        assert 1 <= num_speculative_tokens <= self.num_speculative_tokens, ('),
 ('        return out\n\n    def _sample_sequential(', '        if num_speculative_tokens < self.num_speculative_tokens:\n            out = out[:, :num_speculative_tokens].contiguous()\n        return out\n\n    def _sample_sequential(')
],'STAR_ABC integration')
patch(pkg/'hybrid/w26_dual_graph.py',[
 ('def intercept(wrapper,ctx,args,kwargs):\n    rt=', 'def intercept(wrapper,ctx,args,kwargs):\n    from .abc_runtime import intercept as abc_intercept  # STAR_ABC integration\n    abc = abc_intercept(wrapper,ctx,args,kwargs)\n    if abc is not None:return abc\n    rt=')
],'STAR_ABC integration')
patch(pkg/'hybrid/cumulative_startup.py',[
 ('    worker._w26_attention_initial_receipt=receipt;return receipt','    worker._w26_attention_initial_receipt=receipt\n    if os.environ.get("STAR_ABC","0")=="1":\n        from .abc_startup import prepare as abc_prepare  # STAR_ABC integration\n        receipt["abc_preparation"]=abc_prepare(worker)\n    return receipt')
],'STAR_ABC integration')
patch(pkg/'hybrid/initial_decode_contract.py',[
 ('def initial_decode(runner,consensus,records,b=1):','def initial_decode(runner,consensus,records,b=1,query_len=6):  # STAR_ABC integration'),
 ('    state=dict(runner=runner,consensus=consensus,records=records,calls=0,b=b)','    if query_len not in (4,6):raise RuntimeError("Unsupported initial query length")\n    state=dict(runner=runner,consensus=consensus,records=records,calls=0,b=b,query_len=query_len)'),
 ("        expected=dict(num_tokens=6*b,num_reqs=b,num_tokens_padded=6*b,num_reqs_padded=b,\n            causal=True,max_query_len=6,actual_tokens=6*b,common_reqs=b,query_start=list(range(0,6*b+1,6)),", "        q=state['query_len']\n        expected=dict(num_tokens=q*b,num_reqs=b,num_tokens_padded=q*b,num_reqs_padded=b,\n            causal=True,max_query_len=q,actual_tokens=q*b,common_reqs=b,query_start=list(range(0,q*b+1,q)),")
],'STAR_ABC integration')

# Reuse the same proven attention math with a new explicit 4B/6B profile vocabulary.
s=(pkg/'hybrid/r2_attention.py').read_text(encoding='utf-8')
s=s.replace('class RowsProfile:', 'class Profile:').replace('RowsProfile','Profile').replace('class AttentionStateRows','class State')
s=s.replace('if rows % 6 or not 12 <= rows <= 48:', 'if rows not in {4*b for b in range(1,9)} | {6*b for b in range(1,9)}:')
s=s.replace('return attention_workspace_bytes(self.rows // 6, (self.splits, self.warps, self.direct))','r,s=self.rows,self.splits\n        return (r*s*16*2+r*s*16*512)*4+r*16*512*2+4')
s=s.replace('R2 attention rows must be 6B for B in 2..8','ABC attention rows must be an explicit 4B/6B family')
(pkg/'hybrid/abc_attention.py').write_text('# STAR_ABC row-family extension of the retained R2 math.\n'+s,encoding='utf-8')

runner=pkg/'vllm/v1/worker/gpu_model_runner.py'
patch(runner,[
 ('            r2_rows = getattr(sys.modules.get("hybrid.r2_bank"), "DUMMY_ROWS", ())\n            if not ((num_tokens == 6 or num_tokens in r2_rows) and uniform_query_len == 6 and uniform_decode', '            r2_rows = getattr(sys.modules.get("hybrid.r2_bank"), "DUMMY_ROWS", ())\n            abc_ctx = getattr(sys.modules.get("hybrid.abc_runtime"), "CAPTURE", None)  # STAR_ABC integration\n            abc_shape = abc_ctx is not None and num_tokens == abc_ctx.rows and uniform_query_len == abc_ctx.plan.k + 1\n            if not (((num_tokens == 6 or num_tokens in r2_rows) and uniform_query_len == 6 or abc_shape) and uniform_decode'),
 ('            model_output = self._model_forward(','            from hybrid import abc_gate\n            if abc_gate.MASK is not None:\n                abc_gate.before(self,scheduler_output,cudagraph_mode,batch_desc)\n            model_output = self._model_forward('),
 ('                logits = self.model.compute_logits(sample_hidden_states)\n            else:', '                from hybrid.abc_greedy import logits as abc_logits\n                logits = abc_logits(self,sample_hidden_states,scheduler_output)\n                if logits is None:\n                    logits = self.model.compute_logits(sample_hidden_states)\n            else:')
],'STAR_ABC integration')

rs=pkg/'vllm/v1/sample/rejection_sampler.py'
patch(rs,[
 ('        assert metadata.max_spec_len <= MAX_SPEC_LEN\n','        from hybrid import abc_gate  # STAR_ABC integration\n        metadata = abc_gate.rejection(metadata,sampling_metadata)\n        assert metadata.max_spec_len <= MAX_SPEC_LEN\n'),
 ('        assert metadata.max_spec_len <= MAX_SPEC_LEN\n\n        bonus_logits_indices = metadata.bonus_logits_indices','        assert metadata.max_spec_len <= MAX_SPEC_LEN\n\n        from hybrid.abc_greedy import verify\n        direct = verify(self,metadata,draft_probs,logits,sampling_metadata)\n        if direct is not None:\n            return direct\n        bonus_logits_indices = metadata.bonus_logits_indices')
],'STAR_ABC integration')

sched=pkg/'vllm/v1/core/sched/scheduler.py'
patch(sched,[
 ('from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup','from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup\nfrom hybrid import abc_policy as _abc_policy  # STAR_ABC integration'),
 ('            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n                len(num_scheduled_tokens)\n            ]','            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n                len(num_scheduled_tokens)\n            ]\n        if self.num_spec_tokens > 0 and num_scheduled_tokens and _abc_policy.enabled():\n            num_spec_tokens_to_schedule = _abc_policy.choose(self,num_scheduled_tokens)'),
 ('padded_num_tokens = 1 + self.num_spec_tokens','padded_num_tokens = 1 + _abc_policy.pad_k(self)'),
 ('and num_new_tokens != 1 + self.num_spec_tokens','and num_new_tokens != 1 + _abc_policy.pad_k(self)'),
 ('assert num_new_tokens == 1 + self.num_spec_tokens','assert num_new_tokens == 1 + _abc_policy.pad_k(self)'),
 ('                num_rejected = num_draft_tokens - num_accepted','                num_rejected = num_draft_tokens - num_accepted\n                _abc_policy.observe(self,req_id,num_draft_tokens,num_accepted)'),
 ('        self._dsv41_pp_sent.pop(request.request_id, None)','        self._dsv41_pp_sent.pop(request.request_id, None)\n        _abc_policy.forget(request.request_id)')
],'STAR_ABC integration')
# Placeholder update must use selected width too (same scope as upstream patch).
s=sched.read_text(encoding='utf-8');needle='] * self.num_spec_tokens'
if s.count(needle)==1:sched.write_text(s.replace(needle,'] * _abc_policy.pad_k(self)'),encoding='utf-8')
elif '] * _abc_policy.pad_k(self)' not in s:raise RuntimeError('Unexpected scheduler placeholder multiplicity')

control=root/'host-overlay/resident-api/w9_resident_api/control.py'
patch(control,[
 ('  for name in (\'inbox\',\'responses\',\'completed\'):', '  from hybrid.abc_control import Owner as ABCOwner  # STAR_ABC integration\n  self.abc_bank=ABCOwner(self)\n  for name in (\'inbox\',\'responses\',\'completed\'):'),
 (' async def handle(self,q):\n',' async def handle(self,q):\n  if isinstance(q,dict) and str(q.get("op","")).startswith("abc_"):\n   self.abc_bank.validate(q)\n   async with self.lock:return await self.abc_bank.handle_locked(q)\n')
],'STAR_ABC integration')
worker=root/'host-overlay/integration/w9_exl_ab/worker_extension.py'
methods=''
for name in ['status','prepare','commit','rollback','end','route']:
    args='request' if name in ['prepare','route'] else 'token' if name in ['commit','rollback'] else ''
    methods+=f' def wabc_{name}(self{","+args if args else ""}):\n  from hybrid.abc_control import worker_invoke\n  return worker_invoke(self,{name!r}{","+args if args else ""})\n'
s=worker.read_text(encoding='utf-8')
if 'def wabc_status' not in s:worker.write_text(s+'\n'+methods,encoding='utf-8')
elif 'def wabc_route' not in s:worker.write_text(s+'\n def wabc_route(self,request):\n  from hybrid.abc_control import worker_invoke\n  return worker_invoke(self,"route",request)\n',encoding='utf-8')
fold=root/'host-overlay/integration/w9_exl_ab/fold_serve.py'
patch(fold,[
 ('def _important(args):\n', 'def _important(args):\n    from hybrid.abc_runtime import context as abc_context  # STAR_ABC integration\n    ctx=abc_context()\n    if ctx is not None and ctx.plan.b==1 and ctx.plan.k==5 and ctx.plan.experts!="tc":\n        return None  # preserve the exact old B1 mixed executable via private-node update\n')
],'STAR_ABC integration')

patch(pkg/'hybrid/attention_admission.py',[
 ('            try:return await self.app(scope,receive,send)','            try:\n                from .abc_admission import receive_for  # STAR_ABC integration\n                checked=await receive_for(gate,scope,receive,send)\n                if checked is not None:return await self.app(scope,checked,send)')
],'STAR_ABC integration')

patch(pkg/'hybrid/dsv41_r1_round.py',[
 ('        self._rec = rec\n\n    def before_forward', '        from .abc_runtime import BANK as abc_bank  # STAR_ABC integration\n        if abc_bank is not None and abc_bank.selection is not None:\n            rec["abc"] = dict(profile=abc_bank.selection.selected,identity=abc_bank.selection.identity,epoch=abc_bank.selection.epoch,session=abc_bank.selection.session)\n        self._rec = rec\n\n    def before_forward')
],'STAR_ABC integration')

for p in list(pkg.rglob('*.py'))+list((root/'host-overlay').rglob('*.py')):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
print('All asserted integration patches applied and AST parsed')
