"""Build the isolated native specialization; never imports CUDA runtime APIs."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nvcc',default='nvcc')
    p.add_argument('--torch-include',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    a.out.mkdir(parents=True,exist_ok=True)
    binary=a.out.resolve()/'native_mixed.cubin'
    cmd=[a.nvcc,'--cubin','-arch=sm_80','-std=c++17','-O3','--use_fast_math',
         '-lineinfo','--expt-relaxed-constexpr','-Xptxas=-v','-diag-suppress=177,20012',
         '-I'+str(a.torch_include.resolve()),'-I'+str((a.torch_include/'torch/csrc/api/include').resolve()),
         str(root/'native/instantiate.cu'),'-o',str(binary)]
    if os.name=='nt':cmd+=['-DWIN32_LEAN_AND_MEAN','-DNOMINMAX','-Xcompiler=/Zc:preprocessor']
    subprocess.run(cmd,check=True)
    record={'file':binary.name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'arch':'sm_80','note':'Build identity only; rerun numerical qualification on a changed toolchain.'}
    (a.out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
