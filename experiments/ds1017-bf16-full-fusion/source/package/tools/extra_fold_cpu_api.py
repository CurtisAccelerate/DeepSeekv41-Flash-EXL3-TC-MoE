"""Actual CPU image ABI/source check, with no GPU device exposure."""
from pathlib import Path
import ast,ctypes,hashlib,json,os,re,subprocess,tempfile,unittest
package=Path('/opt/star-prefill/package')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void'
assert not Path('/dev/nvidia0').exists()
import torch
assert not torch.cuda.is_initialized()
fold=package/'host-overlay/integration/w9_exl_ab/fold_serve.py'
assert hashlib.sha256(fold.read_bytes()).hexdigest()=='4fd2ece81de428012f30ddeaa01210260817b10121ef6bef96e29cde0be1fe60'
node=next(n for n in ast.parse(fold.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='GuLaunch')
ns={'ctypes':ctypes};exec(compile(ast.Module(body=[node],type_ignores=[]),str(fold),'exec'),ns)
Gu=ns['GuLaunch'];source=(package/'narrow-native-source/launch.cu').read_text(encoding='utf-8')
struct=re.search(r'struct GuLaunch\s*\{.*?\};',source,re.S).group(0)
fields=re.findall(r'(void\*|int64_t|double)\s+(\w+)\s*;',struct)
assert [name for kind,name in fields]==[name for name,typ in Gu._fields_]
types={'void*':ctypes.c_void_p,'int64_t':ctypes.c_int64,'double':ctypes.c_double}
assert all(types[kind] is typ for (kind,name),(other,typ) in zip(fields,Gu._fields_))
cpp='#include <cstdio>\n#include <cstdint>\n#include <cstddef>\n'+struct+'\nint main(){\n'
cpp+='std::printf("%zu\\n",sizeof(GuLaunch));\n'
for kind,name in fields:cpp+='std::printf("%zu\\n",offsetof(GuLaunch,'+name+'));\n'
cpp+='}\n'
with tempfile.TemporaryDirectory(prefix='narrow-abi-') as tmp:
 binary=Path(tmp)/'layout'
 subprocess.run(['g++','-std=c++17','-x','c++','-','-o',str(binary)],input=cpp,text=True,check=True)
 actual=list(map(int,subprocess.check_output([str(binary)],text=True).splitlines()))
expected=[ctypes.sizeof(Gu)]+[getattr(Gu,name).offset for name,typ in Gu._fields_]
assert actual==expected
pins=json.loads(Path('/opt/star-prefill/native-pins.json').read_text())
for name,pin in pins.items():
 p=package/'host-overlay/integration/fold-narrow'/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==pin
 assert ' gu_launch' in subprocess.check_output(['nm','-D',str(p)],text=True)
import exllamav3_ext as ext
assert callable(ext.exl3_moe_coop) and callable(ext.reconstruct)
doc=ext.exl3_moe_coop.__doc__ or ''
anonymous=set(map(int,re.findall(r'\barg(\d+)\s*:',doc)))
named=re.findall(r'(?:\(|,)\s*([A-Za-z_]\w*)\s*:',doc.split('\n')[0])
assert anonymous==set(range(36)) or len(named)==36,('Actual coop ABI signature differs',doc)
suite=unittest.TestLoader().discover(str(package/'draft-policy-tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun==13 and not result.skipped
assert not torch.cuda.is_initialized()
receipt=dict(passed=True,draft_contract_tests=result.testsRun,gu_launch_fields=len(fields),gu_launch_bytes=ctypes.sizeof(Gu),cpp_ctypes_offsets_exact=True,original_fold_unchanged=True,native_export_and_pins_passed=True,actual_incumbent_coop_arguments=36,actual_incumbent_signature=doc,torch_cuda_initialized=False,gpu_qualified=False)
Path('/opt/star-prefill/extra-fold-cpu-api.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
