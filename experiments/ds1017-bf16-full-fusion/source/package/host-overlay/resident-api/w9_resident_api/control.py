"""Allowlisted local controls. No chat renderer, arbitrary RPC, or engine shutdown.

Native pause(wait, clear_cache=False) queues new engine requests and drains work.
HTTP tests must still have exclusive traffic ownership while resumed.
"""
import asyncio,json,os,re,time,uuid
from pathlib import Path

ARMS=('baseline','transpose')
HASHES={'baseline':'506f7ccd0108693e8c9f97f866330c976d72fbfca8e093e414f2150962d7560f',
        'transpose':'a4c2ed3e83ee8c9025962379fe1782b66c38b340f05cfa986b6d29c5ded63187'}

class Control:
 def __init__(self,engine,root):
  self.engine=engine;self.root=Path(root);self.lock=asyncio.Lock();self.epoch=0
  self.collective_profile='stock'
  self.arm_name='baseline';self.active_versions={a:'boot' for a in ARMS};self.active_hashes=dict(HASHES)
  self.poisoned=False;self.phase='initializing';self.reason=None;self.last_ranks=None
  self.ranks_timestamp=None;self.stale_ranks=True
  self.capture_generations={};self.worker_pids={};self.consumed=set();self.task=None
  self.admission=None
  from hybrid.attention_owner import AttentionOwner
  self.attention_bank=AttentionOwner(self)
  from hybrid.production_control import ProductionOwner
  self.production_bank=ProductionOwner(self)
  from hybrid.abc_control import Owner as ABCOwner  # STAR_ABC integration
  self.abc_bank=ABCOwner(self)
  from hybrid.prefill_control import Owner as PrefillOwner
  self.prefill_bank=PrefillOwner(self)
  for name in ('inbox','responses','completed'):(self.root/name).mkdir(parents=True,exist_ok=True)
 def snapshot(self):
  return dict(owner_pid=os.getpid(),phase=self.phase,reason=self.reason,poisoned=self.poisoned,
              epoch=self.epoch,arm=self.arm_name,collective_profile=self.collective_profile,active_versions=dict(self.active_versions),active_hashes=dict(self.active_hashes),
              ranks=self.last_ranks,ranks_timestamp=self.ranks_timestamp,stale_ranks=self.stale_ranks,timestamp=time.time(),public_admission=bool(self.admission and self.admission.open))
 def save(self,name,value):
  target=self.root/name;tmp=target.with_suffix(target.suffix+'.tmp')
  tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(target)
 def persist(self):
  try:self.save('status.json',self.snapshot())
  except Exception as e:print('W9_API_DIAGNOSTIC_WRITE_FAILED',repr(e),flush=True)
 async def rpc(self,name,args=(),timeout=180):
  rows=await self.engine.collective_rpc(name,timeout=timeout,args=args)
  if not isinstance(rows,list) or len(rows)!=4 or sorted(r['rank'] for r in rows)!=list(range(4)):
   raise RuntimeError('Missing or duplicate worker acknowledgements')
  return rows
 async def pause(self):
  self.phase='pausing';self.persist()
  await asyncio.wait_for(self.engine.pause_generation(mode='wait',clear_cache=False),timeout=300)
  if not await self.engine.is_paused():raise RuntimeError('Native scheduler did not confirm paused state')
  self.phase='paused';self.persist()
 async def quarantine(self,reason,poison=False):
  if self.admission:self.admission.open=False
  self.poisoned|=poison;self.reason=reason
  try:await self.pause()
  except Exception as e:self.reason+='; pause confirmation failed: '+repr(e)
  self.phase='quarantined';self.persist()
 async def qualify(self):
  if self.poisoned:raise RuntimeError('Sticky control poison; no automatic recovery')
  self.stale_ranks=True
  rows=await self.rpc('w9_status',timeout=60)
  self.last_ranks=rows;self.ranks_timestamp=time.time();self.stale_ranks=False
  if any(r['poisoned'] for r in rows):self.poisoned=True
  if os.environ.get('W15_PAIR_TRANSPORT')=='1':
   for r in rows:
    ps=r.get('pair')
    if not isinstance(ps,list) or len(ps)!=1 or not ps[0]['ready'] or ps[0]['cubin_sha256']!='04ddd502f8cd651eb16d336804f0b1a2fc2a93061683f6dccb2ff009b63d1d65':raise RuntimeError('Missing or wrong pair readiness receipt')
    a=ps[0].get('dedicated_arena',{})
    if (a.get('nbytes'),a.get('slot_bytes'),a.get('blocks'),a.get('threads'),a.get('tile'),a.get('rounds'))!=(491520,491520,8,256,1024,1):raise RuntimeError('W19 fixed48 arena contract differs')
    if not a.get('before_pin_exact') or not a.get('after_pin_exact') or ps[0]['device_inbox_bytes']!=1966080 or ps[0]['large_approved_bytes'] or ps[0]['large_slot']:raise RuntimeError('W19 arena placement/memory policy differs')
   if len({r['pair'][0]['dedicated_arena']['plan_sha256'] for r in rows})!=1:raise RuntimeError('W19 placement plans differ')
   if len({json.dumps(r['pair'][0]['large_approved_bytes']) for r in rows})!=1:raise RuntimeError('Large pair policies differ')
  for r in rows:
   if (not r['ready'] or r['poisoned'] or r['pending'] is not None or r['capturing']
       or r['quarantine_reason'] is not None or r['version_pending'] is not None
       or r['hashes']!=HASHES or tuple(r['cc'])!=(8,0) or r['epoch']!=self.epoch
       or r['capture_generation']<1 or r['arm'] not in ARMS
       or not r['graph_sets']['baseline'] or r['graph_sets']['baseline']!=r['graph_sets']['transpose']
       or r['graphs']['baseline']!=r['graphs']['transpose']
       or len(r['graph_sets']['baseline'])!=r['graphs']['baseline']
       or any(r['launches'][a]<=0 for a in ARMS)):
    raise RuntimeError('Worker qualification failed: '+str(r['rank']))
   if r.get('collective',{}).get('profile')!=self.collective_profile or r.get('collective',{}).get('pending') is not None:raise RuntimeError('Collective profile mismatch')
   if set(r['active_versions'])!=set(ARMS) or set(r['active_hashes'])!=set(ARMS):raise RuntimeError('Unexpected active-version keys')
   if any(not isinstance(v,str) or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',v) for v in r['active_versions'].values()):raise RuntimeError('Invalid active-version label')
   if r['arm']!=self.arm_name or r['active_versions']!=self.active_versions or r['active_hashes']!=self.active_hashes:
    raise RuntimeError('Actual worker selection differs from committed controller state')
   if self.worker_pids and (r['pid']!=self.worker_pids[r['rank']] or r['capture_generation']!=self.capture_generations[r['rank']]):
    raise RuntimeError('Worker identity or capture generation changed')
  if len({r['arm'] for r in rows})!=1 or len({json.dumps(r['active_versions'],sort_keys=True) for r in rows})!=1:
   raise RuntimeError('Worker arm/version states disagree')
  if not self.worker_pids:
   self.worker_pids={r['rank']:r['pid'] for r in rows};self.capture_generations={r['rank']:r['capture_generation'] for r in rows}
  return rows
 async def initialize(self):
  # Called after native app state exists. Failures stay inside the owner lifetime.
  try:
   await self.pause();await self.qualify();self.reason=None
   from hybrid.attention_startup import initialize_owner
   await initialize_owner(self)
  except Exception as e:
   from hybrid.attention_startup import UnsafePreparation
   await self.quarantine('Initial qualification failed: '+repr(e),poison=isinstance(e,UnsafePreparation))
  self.persist()
 def validate(self,q):
  if not isinstance(q,dict):raise ValueError('Control request must be object')
  op=q.get('op');allowed={'status':{'op'},'pause':{'op'},'resume':{'op'},'release_serving':{'op'},'requalify':{'op'},
       'collective_profile':{'op','profile'},'select_arm':{'op','arm'},'register_version':{'op','arm','version','manifest','manifest_sha256'}}
  if op not in allowed or set(q)!=allowed[op]:raise ValueError('Unknown operation or fields')
  if op=='collective_profile' and q['profile'] not in ('stock','six','six48','pair_hybrid'):raise ValueError('Unknown collective profile')
  if 'arm' in q and q['arm'] not in ARMS:raise ValueError('Unknown arm')
  if op=='register_version':
   if q['arm']!='transpose' or not isinstance(q['version'],str) or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',q['version']):raise ValueError('Invalid version')
   if not isinstance(q['manifest'],str) or Path(q['manifest']).name!=q['manifest'] or not q['manifest'].endswith('.json'):raise ValueError('Invalid manifest basename')
   if not isinstance(q['manifest_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',q['manifest_sha256']):raise ValueError('Invalid manifest SHA256')
  return op
 async def arm(self,arm):
  await self.qualify();next_epoch=self.epoch+1
  try:
   for name,phase in [('w9_prepare','prepared'),('w9_commit','committed')]:
    rows=await self.rpc(name,(arm,next_epoch),timeout=60)
    for r in rows:
     if (r['arm'],r['epoch'],r['phase'],r['hashes'],r['capture_generation'])!=(arm,next_epoch,phase,HASHES,self.capture_generations[r['rank']]):
      raise RuntimeError('Arm acknowledgement differs')
     if (r['version'],r['version_sha256'])!=(self.active_versions[arm],self.active_hashes[arm]):raise RuntimeError('Active version acknowledgement differs')
   self.epoch=next_epoch;self.arm_name=arm;await self.qualify()
  except Exception as e:
   await self.quarantine('Uncertain arm transaction: '+repr(e),poison=True);raise
 async def version(self,q):
  await self.qualify();token=uuid.uuid4().hex;record={'token':token,'steps':[]}
  try:
   expected_hash=None
   for name,args,phase in [('w9_version_stage',(token,q['arm'],q['version'],q['manifest'],q['manifest_sha256']),'staged'),
                           ('w9_version_capture',(token,),'qualified')]:
    rows=await self.rpc(name,args);record['steps'].append({'method':name,'rows':rows})
    if not all(r['ok'] for r in rows):
     if any(r.get('poisoned') or not r.get('rejected',False) for r in rows if not r['ok']):raise RuntimeError('Unsafe worker failure')
     aborted=await self.rpc('w9_version_abort',(token,));record['abort']=aborted
     if not all(r['ok'] and r['result']['token']==token and r['result']['phase'] in ('aborted','aborted_retained','absent') for r in aborted):raise RuntimeError('Abort acknowledgements uncertain')
     await self.qualify();record['outcome']='rejected_old_version_retained';return record
    for r in rows:
     v=r['result']
     if (v['token'],v['arm'],v['version'],v['phase'])!=(token,q['arm'],q['version'],phase):raise RuntimeError('Version phase acknowledgement differs')
    hashes={r['result']['sha256'] for r in rows}
    if len(hashes)!=1 or (expected_hash is not None and next(iter(hashes))!=expected_hash):raise RuntimeError('Candidate identity changed across ranks/phases')
    expected_hash=next(iter(hashes))
   rows=await self.rpc('w9_version_commit',(token,));record['commit']=rows
   if not all(r['ok'] and (r['result']['token'],r['result']['arm'],r['result']['version'],r['result']['phase'],r['result']['sha256'])==(token,q['arm'],q['version'],'committed',expected_hash) for r in rows):raise RuntimeError('Partial or mismatched version commit')
   self.active_versions[q['arm']]=q['version'];self.active_hashes[q['arm']]=expected_hash
   await self.qualify();record['outcome']='committed';return record
  except Exception as e:
   await self.quarantine('Uncertain version transaction: '+repr(e),poison=True);raise
 async def collective(self,profile):
  await self.qualify();token=uuid.uuid4().hex;record=dict(token=token,profile=profile,steps=[])
  try:
   for method,phase in [('prepare','prepared'),('apply','applied')]:
    rows=await self.rpc('w14_collective_profile',(method,token,profile));record['steps'].append(dict(method=method,rows=rows))
    if not all(r['ok'] for r in rows):
     if any(r.get('poisoned') or not r.get('rejected') for r in rows if not r['ok']):raise RuntimeError('Unsafe collective profile failure')
     aborted=await self.rpc('w14_collective_profile',('abort',token,profile));record['abort']=aborted
     if not all(r['ok'] and r['result']['phase']=='aborted' and r['result']['token']==token and r['result']['profile']==self.collective_profile for r in aborted):raise RuntimeError('Collective rollback uncertain')
     await self.qualify();record['outcome']='rejected_old_profile_retained';return record
    if not all((r['result']['token'],r['result']['profile'],r['result']['phase'])==(token,profile,phase) for r in rows):raise RuntimeError('Collective prepare/apply mismatch')
    if len({json.dumps(r['result']['counts'],sort_keys=True) for r in rows})!=1:raise RuntimeError('Collective eligible-node counts differ across ranks')
   rows=await self.rpc('w14_collective_profile',('commit',token,profile));record['commit']=rows
   if not all(r['ok'] and (r['result']['token'],r['result']['profile'],r['result']['phase'])==(token,profile,'committed') for r in rows):raise RuntimeError('Collective partial commit')
   self.collective_profile=profile;await self.qualify();record['outcome']='committed';return record
  except Exception as e:
   await self.quarantine('Uncertain collective profile transaction: '+repr(e),poison=True);raise

 async def handle(self,q):
  if isinstance(q,dict) and str(q.get('op','')).startswith('prefill_'):
   self.prefill_bank.validate(q)
   async with self.lock:return await self.prefill_bank.handle_locked(q)
  if isinstance(q,dict) and str(q.get("op","")).startswith("abc_"):
   self.abc_bank.validate(q)
   async with self.lock:return await self.abc_bank.handle_locked(q)
  if isinstance(q,dict) and str(q.get('op','')).startswith('production_'):
   self.production_bank.validate(q)
   async with self.lock:return await self.production_bank.handle_locked(q)
  if isinstance(q,dict) and str(q.get('op','')).startswith('attention_'):
   self.attention_bank.validate(q) # Validate before pause/admission changes.
   async with self.lock:
    try:return await self.attention_bank.handle_locked(q)
    except Exception as exc:
     if getattr(self.admission,'attention_guarded',False):
      try:
       await self.attention_bank.abort('attention operation failed: '+repr(exc))
       await self.qualify()
      except Exception as cleanup:
       await self.quarantine('Uncertain attention operation/rollback: '+repr(cleanup),poison=True)
     raise
  op=self.validate(q) # Invalid JSON/schema never changes engine state.
  async with self.lock:
   self.attention_bank.guard_legacy_operation(op)
   if op=='status':
    try:
     self.stale_ranks=True
     self.last_ranks=await self.rpc('w9_status',timeout=60)
     self.ranks_timestamp=time.time();self.stale_ranks=False
     return {'status':self.snapshot(),'scheduler_paused':await self.engine.is_paused()}
    except Exception as e:
     await self.quarantine('Status failed: '+repr(e));raise
   if self.poisoned:raise RuntimeError('Sticky poison; inference must remain paused')
   # Control queue is serialized. Native pause is the only scheduling gate.
   try:
    if self.admission:self.admission.open=False
    await self.pause()
    if op in ('requalify','resume','release_serving'):await self.qualify()
    elif op=='select_arm':await self.arm(q['arm'])
    elif op=='collective_profile':return {'collective_transaction':await self.collective(q['profile']),'status':self.snapshot()}
    elif op=='register_version':return {'version_transaction':await self.version(q),'status':self.snapshot()}
    if op in ('resume','release_serving'):
     await self.engine.resume_generation()
     if await self.engine.is_paused():raise RuntimeError('Resume was not confirmed')
     self.phase='serving';self.reason=None
     if op=='release_serving' and self.admission:self.admission.open=True
    # Arm/version changes intentionally remain paused until explicit resume.
    self.persist();return {'status':self.snapshot()}
   except Exception as e:
    if self.phase=='pausing':
     self.phase='quarantined';self.reason='Pause failed before mutation: '+repr(e);self.persist()
    elif self.phase!='quarantined':await self.quarantine('Control failed: '+repr(e))
    raise
 async def process(self,path):
  rid=path.stem
  if rid in self.consumed or (self.root/'completed'/(rid+'.done')).exists() or (self.root/'responses'/path.name).exists():
   path.replace(self.root/'completed'/(rid+'.duplicate'));return
  if len(self.consumed)>=100000:
   await self.quarantine('Control ID capacity reached');raise RuntimeError('Control ID capacity')
  self.consumed.add(rid);active=path.with_suffix('.running');path.rename(active)
  result={'id':rid,'started':time.time()}
  try:
   if active.stat().st_size>65536:raise ValueError('Control request too large')
   result.update(await self.handle(json.loads(active.read_text())));result['ok']=True
  except Exception as e:result.update(ok=False,error=repr(e),status=self.snapshot())
  try:self.save('responses/'+path.name,result)
  except Exception as e:print('W9_API_RESPONSE_WRITE_FAILED',repr(e),flush=True)
  try:active.replace(self.root/'completed'/(rid+'.done'))
  except Exception as e:print('W9_API_ARCHIVE_FAILED',repr(e),flush=True)
  self.persist()
 async def run(self):
  while True:
   try:
    for path in sorted((self.root/'inbox').glob('*.json')):
     try:await self.process(path)
     except Exception as e:print('W9_API_JOB_FAILED',repr(e),flush=True)
    await asyncio.sleep(.2)
   except asyncio.CancelledError:raise
   except Exception as e:print('W9_API_QUEUE_FAILED',repr(e),flush=True);await asyncio.sleep(1)
