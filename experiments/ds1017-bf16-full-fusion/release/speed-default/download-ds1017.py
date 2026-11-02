"""Download verified Docker image parts from the official project release."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,urllib.request

RELEASE='https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/releases/download/2026.11.02-speed-k5'
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fetch(name,path):
 assert Path(name).name==name and '/' not in name and '\\' not in name
 temporary=path.with_name(path.name+'.partial')
 with urllib.request.urlopen(RELEASE+'/'+name,timeout=120) as response,temporary.open('wb') as out:shutil.copyfileobj(response,out,1024**2)
 temporary.replace(path)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--directory',type=Path,default=Path('ds1017-download'))
 parser.add_argument('--load',action='store_true',help='Load the verified stream into the current Docker daemon and apply its versioned tag.')
 parser.add_argument('--join',action='store_true',help='Also write one joined .docker.tar.gz file.')
 args=parser.parse_args();root=args.directory.resolve();root.mkdir(parents=True,exist_ok=True)
 manifest_path=root/'DOWNLOAD-MANIFEST.json';fetch(manifest_path.name,manifest_path)
 manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert manifest['format']=='docker-save-gzip-split-v1'
 archive=hashlib.sha256();total=0
 for row in manifest['parts']:
  path=root/row['name']
  if not path.exists() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
   print('Downloading',row['name'],flush=True);fetch(row['name'],path)
  assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Part checksum mismatch: '+row['name']
  with path.open('rb') as stream:
   while chunk:=stream.read(4*1024**2):archive.update(chunk);total+=len(chunk)
 assert total==manifest['archive_bytes'] and archive.hexdigest()==manifest['archive_sha256'],'Archive checksum mismatch'
 print('All parts and full archive SHA256 verified.',flush=True)
 if args.join:
  joined=root/manifest['archive_name']
  with joined.open('wb') as out:
   for row in manifest['parts']:
    with (root/row['name']).open('rb') as stream:shutil.copyfileobj(stream,out,4*1024**2)
  print('Joined archive:',joined,flush=True)
 if args.load:
  process=subprocess.Popen(['docker','load'],stdin=subprocess.PIPE)
  try:
   for row in manifest['parts']:
    with (root/row['name']).open('rb') as stream:shutil.copyfileobj(stream,process.stdin,4*1024**2)
   process.stdin.close()
  except BaseException:
   process.kill();raise
  assert process.wait()==0,'docker load failed'
  loaded=json.loads(subprocess.check_output(['docker','image','inspect',manifest['image_id']],text=True))[0]
  assert loaded['Id']==manifest['image_id'],'Loaded image identity mismatch'
  subprocess.run(['docker','tag',manifest['image_id'],manifest['tag']],check=True)
  print('Loaded and verified:',manifest['tag'],manifest['image_id'],flush=True)
 else:print('Use --load to stream the verified image into Docker; --join writes a single archive.')
if __name__=='__main__':main()
