from pathlib import Path
import hashlib,json,os,runpy,sys,unittest
os.environ['STAR_ABC']='1';os.environ['STAR_ABC_KV_DTYPE']='bfloat16'
root=Path('/opt/star-release');publication=root/'publication';package=Path('/opt/star-prefill/package')
assert json.loads((root/'DEFAULT-RECIPE.json').read_text())['status']=='selected_after_final_evidence_review'
runpy.run_path(str(publication/'check_sources.py'),run_name='__main__')
from hybrid.abc_startup import attest
identity=attest()
sys.path.insert(0,str(package/'tools'))
runpy.run_path(str(package/'tools/prefill_cpu_check.py'),run_name='__main__')
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(package/'target-isolation-tests')));assert result.wasSuccessful()
runpy.run_path(str(package/'tools/extra_fold_cpu_api.py'),run_name='__main__')
runpy.run_path(str(package/'tools/proposal_policy_cpu_check.py'),run_name='__main__')
import torch
assert not torch.cuda.is_initialized()
receipt=dict(passed=True,source_manifest_sha256=hashlib.sha256((publication/'SOURCE-MANIFEST.json').read_bytes()).hexdigest(),recipe_sha256=hashlib.sha256((root/'DEFAULT-RECIPE.json').read_bytes()).hexdigest(),identity=identity,cuda_initialized=False,qualified_runtime_base='sha256:7ecbcf725f31d1b5bc05fc6863a16a94ef9c1c6cf7441e0a5c257fcd430e4d94',runtime_and_native_unchanged=True,new_image_gpu_boot_tested=False,scope='Exact-source/native attestation plus CPU tests, derivative packaging only. GPU qualification and benchmarks belong to pinned base/mounted runtime. No new GPU model reload.')
(root/'BUILD-VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print('STAR_RELEASE_VERIFIED '+json.dumps(receipt))
