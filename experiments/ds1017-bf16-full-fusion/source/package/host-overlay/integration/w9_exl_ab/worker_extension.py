"""Named vLLM worker extension; no callable serialization or worker constructor changes."""
class WorkerExtension:
 def w47_production_status(self):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'status')
 def w47_production_prepare(self,request):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'prepare',request)
 def w47_production_commit(self,token):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'commit',token)
 def w47_production_abort(self,reason):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'abort',reason)
 def w9_status(self):
  import torch
  from w9_exl_ab import runtime as R
  from w9_exl_ab.collective_profiles import status as collective_status
  from w9_exl_ab.pair_transport import check_errors, status as pair_status
  try:check_errors(force=True)
  except Exception:
   R.S.poisoned=True;R.S.ready=False
  return dict(pair=pair_status(),collective=collective_status(),pid=__import__("os").getpid(),pending=R.S.pending,capturing=R.S.capturing,quarantine_reason=R.S.quarantine_reason,active_versions=dict(R.S.active_versions),active_hashes=dict(R.S.active_hashes),version_pending=R.S.version_pending,graph_sets={a:sorted(repr(v) for v in R.S.graphs[a]) for a in R.ARMS},rank=int(self.rank),ready=R.S.ready,arm=R.S.arm,epoch=R.S.epoch,poisoned=R.S.poisoned,capture_generation=R.S.capture_generation,graphs={a:len(v) for a,v in R.S.graphs.items()},launches=R.S.launches,graph_budget=R.S.graph_budget,graph_used=R.S.graph_used,graph_receipts=R.S.graph_receipts,device=str(R.S.backend.device),cc=R.S.backend.cc,hashes=R.HASHES,free_total=torch.cuda.mem_get_info(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())

 def w9_prepare(self,arm,epoch):
  from .runtime import rpc_prepare
  return rpc_prepare(self,arm,epoch)

 def w9_commit(self,arm,epoch):
  from .runtime import rpc_commit
  return rpc_commit(self,arm,epoch)

 def w9_version_stage(self,token,arm,version,manifest_name,manifest_sha):
  from .versions import invoke
  return invoke(self,'stage',token,arm,version,manifest_name,manifest_sha)
 def w9_version_capture(self,token):
  from .versions import invoke
  return invoke(self,'capture',token)
 def w9_version_commit(self,token):
  from .versions import invoke
  return invoke(self,'commit',token)
 def w9_version_abort(self,token):
  from .versions import invoke
  return invoke(self,'abort',token)
 def w9_version_status(self):
  from .versions import invoke
  return invoke(self,'status')

 def w14_collective_profile(self,method,token,profile):
  from .collective_profiles import invoke
  return invoke(self,method,token,profile)

 def w26_attention_bank_status(self):
  from hybrid.attention_worker import invoke
  return invoke(self,'status')

 def w26_attention_bank_prepare(self,request):
  from hybrid.attention_worker import invoke
  return invoke(self,'prepare',request)

 def w26_attention_bank_commit(self,token):
  from hybrid.attention_worker import invoke
  return invoke(self,'commit',token)

 def w26_attention_bank_abort(self,reason):
  from hybrid.attention_worker import invoke
  return invoke(self,'abort',reason)

 def w26_attention_benchmark_arm(self,session,identity_sha256):
  from hybrid.attention_worker import invoke
  return invoke(self,'arm',session,identity_sha256)

 def w26_attention_initial_prepare(self,owner_id):
  from hybrid.attention_startup import prepare_worker
  return prepare_worker(self,owner_id)

 def wabc_status(self):
  from hybrid.abc_control import worker_invoke
  return worker_invoke(self,'status')
 def wabc_prepare(self,request):
  from hybrid.abc_control import worker_invoke
  return worker_invoke(self,'prepare',request)
 def wabc_commit(self,token):
  from hybrid.abc_control import worker_invoke
  return worker_invoke(self,'commit',token)
 def wabc_rollback(self,token):
  from hybrid.abc_control import worker_invoke
  return worker_invoke(self,'rollback',token)
 def wabc_end(self):
  from hybrid.abc_control import worker_invoke
  return worker_invoke(self,'end')

 def wabc_route(self,request):
  from hybrid.abc_control import worker_invoke
  return worker_invoke(self,"route",request)

 def wpf_status(self):
  from hybrid.prefill_control import worker_invoke
  return worker_invoke(self,'status')

 def wpf_prepare(self,request):
  from hybrid.prefill_control import worker_invoke
  return worker_invoke(self,'prepare',request)

 def wpf_commit(self,token):
  from hybrid.prefill_control import worker_invoke
  return worker_invoke(self,'commit',token)
