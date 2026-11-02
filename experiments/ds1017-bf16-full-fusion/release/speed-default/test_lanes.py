"""Configuration, guards and real-profile integration; CPU only."""
from pathlib import Path
from unittest.mock import patch
import copy,contextlib,io,json,os,tempfile,unittest
import lane_client,lane_entrypoint
from configure_launch import configure
from lanes import config,environment,validate_command
COMMAND=['python3','/w26-preflight/preflight_launch.py','python3','-m','w9_resident_api.launch','--kv-cache-dtype','bfloat16','--speculative-config','{"method":"dspark","num_speculative_tokens":5}']
def fixture(lane='speed'):
 c=config(lane);identity='test-qualified-bank'
 rows=[dict(rank=i,ok=True,identity=identity,pending=None,available=[c['profile'],'A']) for i in range(4)]
 receipts=[dict(ready=True,identity=identity,capture_scope=c['capture_scope'],kv_cache_dtype='bfloat16',candidate_batch_sizes=list(range(1,9)),candidate_widths=[5,3],memory=dict(free=1073741824,growth=1152*1024**2),extra_fold_checks=dict(passed=True,cases=96),sampler_interleave=dict(passed=True,cases=32),prefill_preparation=dict(ready=True),available=[c['profile'],'A']) for _ in range(4)]
 return dict(kv_cache_dtype='bfloat16',env=environment(lane)),rows,receipts
class LaneTests(unittest.TestCase):
 def test_speed_default(self):self.assertEqual(config()['profile'],'GN-SPEED-WIDE');self.assertEqual(environment()['STAR_ABC_SCOPE'],'draft-speed')
 def test_bf16_actual_scope(self):self.assertEqual(config('bf16')['profile'],'GN-BALANCED-WIDE');self.assertEqual(environment('bf16')['STAR_ABC_SCOPE'],'draft-balanced')
 def test_quality_reference(self):self.assertEqual(config('quality')['profile'],'A');self.assertEqual(config('quality')['prefill'],'REFERENCE')
 def test_invalid_lane_rejected(self):
  with self.assertRaises(ValueError):config('typo')
 def test_cap_not_relaxed(self):
  with self.assertRaises(ValueError):environment('speed',{'STAR_ABC_CAPTURE_MIB':'2048'})
 def test_fp8_cache_not_silently_selected(self):
  with self.assertRaises(ValueError):environment('speed',{'STAR_ABC_KV_DTYPE':'fp8_ds_mla'})
 def test_command_sampling_and_model_unchanged(self):self.assertEqual(validate_command(COMMAND),COMMAND)
 def test_wrong_server_cache_rejected(self):
  c=COMMAND.copy();c[c.index('bfloat16')]='fp8_ds_mla'
  with self.assertRaises(ValueError):validate_command(c)
 def test_wrong_draft_width_rejected(self):
  c=COMMAND.copy();c[-1]='{"num_speculative_tokens":3}'
  with self.assertRaises(ValueError):validate_command(c)
 def test_configure_preserves_mounts_and_preflight(self):
  original=dict(schema='star.abc.launch.v1',kv_cache_dtype='bfloat16',entrypoint=['python3'],cmd=COMMAND[1:],env={'STAR_ABC_CAPTURE_MIB':'1152'},mounts=[dict(Target='/w9/state',Source='/fresh-stage/state')])
  frozen=copy.deepcopy(original);out=configure(original,'speed','sha256:'+'a'*64)
  self.assertEqual(original,frozen);self.assertEqual(out['cmd'],COMMAND);self.assertEqual(out['mounts'],original['mounts']);self.assertTrue(out['requires_new_gpu_qualification']);self.assertEqual(out['entrypoint'][-4:],['--lane','speed','serve','--'])
 def test_repeated_configuration_rejected(self):
  s=dict(schema='star.abc.launch.v1',kv_cache_dtype='bfloat16',entrypoint=['python3','/opt/star-release/lane_entrypoint.py'],cmd=[],env={})
  with self.assertRaises(ValueError):configure(s,'speed','sha256:'+'a'*64)
 def test_all_qualified_lanes(self):
  for name in ('speed','bf16','quality'):self.assertEqual(lane_client.validate_ready(*fixture(name),name),'test-qualified-bank')
 def test_scope_mismatch_never_falls_back(self):
  with self.assertRaises(ValueError):lane_client.validate_ready(*fixture('bf16'),'speed')
 def test_quality_works_in_speed_bank(self):
  spec,rows,rs=fixture('speed');self.assertEqual(lane_client.validate_ready(spec,rows,rs,'quality'),'test-qualified-bank')
 def test_missing_rank_rejected(self):
  s,r,x=fixture()
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r[:-1],x,'speed')
 def test_identity_drift_rejected(self):
  s,r,x=fixture();r[3]['identity']='different'
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_missing_profile_rejected(self):
  s,r,x=fixture();r[2]['available']=['A']
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_stale_startup_rejected(self):
  s,r,x=fixture();x[1]['identity']='old'
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_missing_native_proof_rejected(self):
  s,r,x=fixture();x[0]['extra_fold_checks']['passed']=False
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_sampler_proof_rejected(self):
  s,r,x=fixture();x[1]['sampler_interleave']['cases']=31
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_reference_fallback_not_candidate(self):
  s,r,x=fixture();x[2]['prefill_preparation']['ready']=False
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_floor_failure_rejected(self):
  s,r,x=fixture();x[2]['memory']['free']-=1
  with self.assertRaises(ValueError):lane_client.validate_ready(s,r,x,'speed')
 def test_entrypoint_executes_real_scope(self):
  for lane,scope in [('speed','draft-speed'),('bf16','draft-balanced')]:
   with patch.dict(os.environ,{},clear=True),patch('sys.argv',['lane_entrypoint.py','--lane',lane,'serve','--',*COMMAND]),patch('os.execvpe') as ex:
    lane_entrypoint.main();self.assertEqual(ex.call_args.args[0],'python3');self.assertEqual(ex.call_args.args[1],COMMAND);self.assertEqual(ex.call_args.args[2]['STAR_ABC_SCOPE'],scope)
 def test_describe_defaults_speed(self):
  with patch.dict(os.environ,{},clear=True),patch('sys.argv',['lane_entrypoint.py']),contextlib.redirect_stdout(io.StringIO()) as out:lane_entrypoint.main()
  self.assertEqual(json.loads(out.getvalue())['selected']['lane'],'speed')
 def test_client_selects_requested_profile_and_closes(self):
  for lane in ('speed','bf16','quality'):
   spec,rows,rs=fixture(lane)
   with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);(root/'round-logs').mkdir()
    for i,r in enumerate(rs):(root/'round-logs'/f'abc-preparation-rank{i}.json').write_text(json.dumps(r))
    class Client:
     def __init__(self):self.root=root;self.spec=spec;self.base_url='http://127.0.0.1:1';self.calls=[]
     def headers(self,session):return {}
     def call(self,b):
      self.calls.append(b)
      return {'abc':rows} if b['op']=='abc_status' else dict(identity='test-qualified-bank',prefix_cache_reset=True)
    c=Client();body=dict(stream=True,temperature=1,seed=1017)
    def measure(url,b,**kwargs):self.assertEqual(b,body);return dict(transport_complete=True,http_status=200,request_body=b)
    result=lane_client.execute(c,[body],measure,lane);self.assertEqual(result['lane']['profile'],config(lane)['profile']);self.assertEqual(next(x['profile'] for x in c.calls if x['op']=='abc_select'),config(lane)['profile']);self.assertEqual(c.calls[-1]['op'],'abc_end')
if __name__=='__main__':unittest.main()
