"""Read-only host conditions accompanying every measurement arm."""
import csv,io,subprocess,time

def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,power.limit,temperature.gpu,clocks.sm,clocks.mem,memory.used','--format=csv,noheader,nounits'],text=True)
    rows=[dict(zip(('uuid','name','power_w','temperature_c','sm_mhz','memory_mhz','used_mib'),r)) for r in csv.reader(io.StringIO(raw),skipinitialspace=True)]
    if len(rows)!=4 or any(abs(float(r['power_w'])-140)>0.1 for r in rows):raise RuntimeError('This measured policy requires all four STAR GPUs at 140W')
    return dict(time=time.time(),hardware='STAR-140W',gpus=rows)
