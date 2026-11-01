"""CPU transcription of the pinned native host-only query; confirm against built op before use."""
import json
from pathlib import Path

def size(rows,width):
    def sequence(n):
        a=n*6;p=2 if n<=1024 else 1
        tiles=min(a,a//64+min(384,a)+1)
        ints=4+(a+3)//4*4+4*tiles+a
        return a*p*5120+a*2*width+(a*p+3)//4*4+(ints+3)//4*4
    return max(sequence(min(rows,2048)),sequence(min(rows,1024)))

if __name__=="__main__":
    out=Path(__file__).with_name("WORKSPACE-SOURCE-CALCULATION.json")
    rows=[dict(intermediate=i,max_rows=r,elements=size(r,i),MiB=size(r,i)*4/1024**2)
          for i in (512,640) for r in (65,128,256,512,1024,2048)]
    out.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2)+"\n",encoding="utf-8")
