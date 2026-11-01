"""Exact-byte, immutable shared backing. No torch import or CUDA access.

Production root must be dedicated tmpfs; test callers may explicitly permit disk.
Every attach hashes the existing payload: zero recopy, not zero memory traffic.
The manifest is an externally pinned identity, never a name-only cache key.
"""
import hashlib, json, math, mmap, os, struct
from pathlib import Path

BLOCK = 1024 * 1024

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(BLOCK), b''): h.update(b)
    return h.hexdigest()

def config_identity(path):
    config=json.loads(Path(path).read_text())
    config=config.get('text_config',config)
    return {k:v for k,v in config.items() if k.startswith('engram_')}

def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= n <= 64 * 1024**2: raise ValueError('invalid safetensors header')
        raw = f.read(n)
    if len(raw) != n: raise ValueError('truncated header')
    return json.loads(raw), 8+n, digest(raw)

def make_plan(index_path, config_path, tensor_names, head_sizes, tp=4, dp=1):
    """head_sizes is {layer prefix: exact ordered prime bucket sizes}.

    tensor_names maps each layer prefix to {'weight': checkpoint key,
    'weight_scale_inv': checkpoint key}; caller must use the loaded config's
    EngramLayout values, not approximate equal-row partitions.
    """
    index_path, config_path = Path(index_path), Path(config_path)
    index = json.loads(index_path.read_text())['weight_map']
    plan = {'schema': 1, 'index_sha256': hash_file(index_path),
            'config_sha256': hash_file(config_path), 'tp': tp, 'dp': dp,
            'layout': head_sizes, 'entries': []}
    plan['engram_config']=config_identity(config_path)
    for layer, names in sorted(tensor_names.items()):
        sizes = head_sizes[layer]; local_heads = math.ceil(len(sizes)/(tp*dp))
        if (tp*dp-1)*local_heads >= len(sizes): raise ValueError('empty rank')
        for kind, key in sorted(names.items()):
            path = (index_path.parent/index[key]).resolve()
            hdr, base, hdr_hash = header(path); t = hdr[key]
            if len(t['shape']) != 2: raise ValueError('expected matrix')
            if t['dtype'] not in ('F8_E4M3', 'F8_E8M0', 'U8'): raise ValueError(t['dtype'])
            a,b=t['data_offsets']; rows,width=t['shape']
            if b-a != rows*width or sum(sizes)>rows: raise ValueError('layout/size mismatch')
            for rank in range(tp*dp):
                start=sum(sizes[:rank*local_heads]); end=sum(sizes[:(rank+1)*local_heads])
                plan['entries'].append(dict(layer=layer,kind=kind,rank=rank,
                    checkpoint_key=key,source=str(path),source_size=path.stat().st_size,
                    source_mtime_ns=path.stat().st_mtime_ns,
                    source_ctime_ns=path.stat().st_ctime_ns,source_dev=path.stat().st_dev,
                    source_ino=path.stat().st_ino,
                    header_sha256=hdr_hash,dtype=t['dtype'],full_shape=t['shape'],
                    shape=[end-start,width],stride=[width,1],row_start=start,row_end=end,
                    offset=base+a+start*width,nbytes=(end-start)*width))
    return plan

def source_identity(st):
    return (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_dev,st.st_ino)

def expected_source_identity(e):
    return tuple(e[k] for k in ('source_size','source_mtime_ns','source_ctime_ns','source_dev','source_ino'))

def verify_selection(manifest,index_path,config_path):
    """Only Engram membership/layout is identity; whole model hashes are provenance."""
    index_path=Path(index_path);wm=json.loads(index_path.read_text())['weight_map']
    cfg=config_identity(config_path)
    if cfg!=manifest['engram_config']:raise ValueError('Engram config changed')
    seen=set()
    for e in manifest['entries']:
        key=e['checkpoint_key']
        if key in seen:continue
        seen.add(key)
        if key not in wm:raise ValueError('Engram tensor removed')
        selected=(index_path.parent/wm[key]).resolve()
        if selected!=Path(e['source']).resolve():raise ValueError('Engram source mapping changed')
        if source_identity(selected.stat())!=expected_source_identity(e):raise ValueError('Engram source identity changed')
        if header(selected)[2]!=e['header_sha256']:raise ValueError('Engram header changed')

def capacity_check(plan,available_ram,available_storage,memlock,ram_floor=64*1024**3,storage_floor=8*1024**3):
    rounded=sum((e['nbytes']+4095)//4096*4096 for e in plan['entries'])
    ranks={}
    for e in plan['entries']:ranks[e['rank']]=ranks.get(e['rank'],0)+(e['nbytes']+4095)//4096*4096
    if available_ram-rounded<ram_floor:raise RuntimeError('insufficient RAM headroom for backing')
    if available_storage-rounded<storage_floor:raise RuntimeError('insufficient tmpfs capacity/headroom')
    if memlock!=-1 and memlock<max(ranks.values()):raise RuntimeError('memlock limit below maximum rank registration')
    return dict(rounded_bytes=rounded,max_rank_bytes=max(ranks.values()),available_ram=available_ram,
                available_storage=available_storage,memlock=memlock,ram_floor=ram_floor,storage_floor=storage_floor)

def preflight(plan,root):
    import resource
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    fs=os.statvfs(root);limit=resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return capacity_check(plan,info['MemAvailable'],fs.f_bavail*fs.f_frsize,limit)

def ensure_tmpfs(path):
    if os.name != 'posix': raise RuntimeError('production backing requires Linux tmpfs')
    import ctypes as C
    # statfs buffer deliberately overallocated; f_type is the first native long.
    buf=C.create_string_buffer(256); libc=C.CDLL(None,use_errno=True)
    if libc.statfs(os.fsencode(path),C.byref(buf)): raise OSError(C.get_errno(),'statfs')
    if C.c_long.from_buffer(buf).value != 0x01021994: raise RuntimeError('backing is not tmpfs')

def prepare(plan, root, *, test_disk=False):
    if sum(e['nbytes'] for e in plan['entries'])>64*1024**2:
        raise RuntimeError('serial builder tiny-only; use NUMA-aware builder for full backing')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    os.chmod(root,0o700)
    if not test_disk: ensure_tmpfs(root)
    allocation_gate=preflight(plan,root) if not test_disk else {'test_only':True}
    # No replacement/deletion of an existing generation. An incomplete directory
    # is intentionally not attachable until its final manifest is published.
    if any(root.iterdir()): raise FileExistsError('use a new empty generation directory')
    entries=[]
    for i,e0 in enumerate(plan['entries']):
        e=dict(e0); source=Path(e['source'])
        before=source.stat()
        if source_identity(before)!=expected_source_identity(e): raise ValueError('source changed')
        _,_,hh=header(source)
        if hh!=e['header_sha256']: raise ValueError('source header changed')
        target=root/f'{i:03d}.bin'; h=hashlib.sha256(); left=e['nbytes']
        with source.open('rb') as src, target.open('xb') as dst:
            src.seek(e['offset'])
            while left:
                b=src.read(min(left,BLOCK))
                if not b: raise ValueError('source truncated')
                dst.write(b);h.update(b);left-=len(b)
            dst.flush();os.fsync(dst.fileno())
        os.chmod(target,0o600)
        after=source.stat()
        if source_identity(before)!=source_identity(after): raise ValueError('source changed during fill')
        e.update(file=target.name,payload_sha256=h.hexdigest())
        entries.append(e)
    body={**plan,'entries':entries,'storage':'tmpfs' if not test_disk else 'test-disk'}
    generation=digest(canonical(body)); manifest={**body,'generation':generation}
    manifest_path=root/'manifest.json';manifest_path.write_bytes(canonical(manifest))
    os.chmod(manifest_path,0o400)
    return {'generation':generation,'manifest_sha256':hash_file(manifest_path),'allocation_gate':allocation_gate,
            'payload_bytes':sum(e['nbytes'] for e in entries)}

class Mapping:
    def __init__(self, root, manifest_sha256, layer, kind, rank, *, test_disk=False,mutable_shared=False):
        root=Path(root);path=root/'manifest.json'
        if hash_file(path)!=manifest_sha256: raise ValueError('manifest identity mismatch')
        manifest=json.loads(path.read_text()); body={k:v for k,v in manifest.items() if k!='generation'}
        if digest(canonical(body))!=manifest['generation']: raise ValueError('generation mismatch')
        if not test_disk:
            ensure_tmpfs(root)
            if manifest['storage']!='tmpfs': raise ValueError('test backing forbidden')
        selected=[e for e in manifest['entries'] if (e['layer'],e['kind'],e['rank'])==(layer,kind,rank)]
        if len(selected)!=1: raise ValueError('missing/duplicate shard')
        self.entry=selected[0];self.manifest=manifest
        source=Path(self.entry['source']);st=source.stat()
        if source_identity(st)!=expected_source_identity(self.entry): raise ValueError('checkpoint changed since fill')
        if header(source)[2]!=self.entry['header_sha256']:raise ValueError('checkpoint header changed')
        filename=self.entry['file']
        if Path(filename).name!=filename: raise ValueError('invalid payload path')
        self.mutable_shared=mutable_shared
        self.file=open(root/filename,'r+b' if mutable_shared else 'rb')
        if manifest.get('payload_layout')=='checkpoint-slices-v1':
            offset=self.entry['payload_offset'];size=self.entry['nbytes']
            if offset!=self.entry['offset'] or offset<0 or size<=0:raise ValueError('slice metadata mismatch')
            if os.fstat(self.file.fileno()).st_size!=self.entry['source_size'] or offset+size>self.entry['source_size']:raise ValueError('slice extent mismatch')
            # Read-only source selection and writable payload alias must resolve
            # to the identical inode; no replacement/copy or alternate payload.
            actual=os.fstat(self.file.fileno())
            if (actual.st_dev,actual.st_ino,actual.st_size)!=(st.st_dev,st.st_ino,st.st_size):
                self.file.close();raise ValueError('payload alias is not the selected checkpoint')
            start=offset//mmap.PAGESIZE*mmap.PAGESIZE;delta=offset-start
            self._region=mmap.mmap(self.file.fileno(),delta+size,offset=start,
                access=mmap.ACCESS_WRITE if mutable_shared else mmap.ACCESS_READ)
            self.buffer=memoryview(self._region)[delta:delta+size]
            if mutable_shared:
                import ctypes as C
                self.registration_address=C.addressof(C.c_char.from_buffer(self._region))
                self.registration_nbytes=len(self._region)
        else:
            if os.fstat(self.file.fileno()).st_size!=self.entry['nbytes']: raise ValueError('payload size mismatch')
            self.buffer=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_WRITE if mutable_shared else mmap.ACCESS_READ)
        # Hash the same mapping subsequently registered, not a separately opened pathname.
        h=hashlib.sha256()
        for i in range(0,len(self.buffer),BLOCK):h.update(self.buffer[i:i+BLOCK])
        if h.hexdigest()!=self.entry['payload_sha256']:
            self.close();raise ValueError('payload corruption')
        if source_identity(source.stat())!=expected_source_identity(self.entry):
            self.close();raise ValueError('checkpoint changed during payload verification')
    def close(self):
        if hasattr(self,'_region'):
            self.buffer.release();self._region.close()
        else:self.buffer.close()
        self.file.close()
