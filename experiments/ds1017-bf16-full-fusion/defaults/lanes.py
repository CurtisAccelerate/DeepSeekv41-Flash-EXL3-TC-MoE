"""Release choices mapped to existing runtime profiles, not new kernel paths."""
import copy,hashlib,json
RUNTIME_BASE='sha256:7ecbcf725f31d1b5bc05fc6863a16a94ef9c1c6cf7441e0a5c257fcd430e4d94'
NATIVE_IDENTITY='738ae435bf007ca27df22f43254c92903f4f9cb2ac3b6e67171a0f556be01877'
LANES={
 'speed':dict(name='Speed',profile='GN-SPEED-WIDE',capture_scope='draft-speed',family=[True,True,'hybrid'],prefill='MOE_RAGGED',output_projection='fp8-short-rows',description='Speed default: short-row FP8 dense/output projection, retained specialized experts, fast draft head/sampler, fixedfive and gateoff.'),
 'bf16':dict(name='BF16 projection',profile='GN-BALANCED-WIDE',capture_scope='draft-balanced',family=[True,False,'hybrid'],prefill='MOE_RAGGED',output_projection='bfloat16',description='BF16 output projection with the retained fast dense/expert/draft paths. Precision-conservative option, not a proved accuracy winner.'),
 'quality':dict(name='Original quality reference',profile='A',capture_scope='draft-balanced',family=[False,False,'ours'],prefill='REFERENCE',output_projection='bfloat16',description='Original target/reference prefill. Numerical leader in latest small teacher fixture, not a general agent-quality guarantee.'),
}
def config(name='speed'):
 if name not in LANES:raise ValueError('Choose speed, bf16 or quality')
 return dict(lane=name,kv_cache_dtype='bfloat16',verification_width=5,confidence_gate=False,adaptive_width=False,proposal_policy='existing-serving-split',**copy.deepcopy(LANES[name]))
def environment(name='speed',base=None):
 c=config(name);env=dict(base or {})
 if env.get('STAR_ABC_CAPTURE_MIB','1152')!='1152':raise ValueError('Retain the measured1152MiB capture cap')
 if env.get('STAR_ABC_KV_DTYPE','bfloat16')!='bfloat16':raise ValueError('These release lanes require BF16 KV')
 env.update(DS1017_LANE=name,STAR_ABC='1',STAR_ABC_SCOPE=c['capture_scope'],STAR_ABC_KV_DTYPE='bfloat16',STAR_ABC_CAPTURE_MIB='1152',STAR_ABC_OPERATOR_CHECKS='1',STAR_PREFILL='1')
 return env
def validate_command(command):
 if not command:raise ValueError('Pass the existing preflight/server command after --')
 if '--kv-cache-dtype' not in command or command[command.index('--kv-cache-dtype')+1]!='bfloat16':raise ValueError('Server command must explicitly use --kv-cache-dtype bfloat16')
 if '--speculative-config' not in command:raise ValueError('Keep the existing fixedfive speculative configuration')
 spec=json.loads(command[command.index('--speculative-config')+1])
 if spec.get('num_speculative_tokens')!=5:raise ValueError('This recipe requires five proposal tokens')
 return list(command)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
