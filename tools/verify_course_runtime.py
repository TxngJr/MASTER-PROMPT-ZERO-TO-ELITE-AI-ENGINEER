#!/usr/bin/env python3
"""Fresh runtime evidence; source hashes prevent silently reusing stale results."""
import os,sys,json,subprocess,time,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def digest(folder):
 h=hashlib.sha256()
 for sub in ['src','tests']:
  for p in sorted((folder/sub).rglob('*.py')):
   h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()
def main():
 import importlib.metadata
 packages={name:importlib.metadata.version(name)for name in ['numpy','torch','tensorflow','pytest','scikit-learn']}
 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TF_CPP_MIN_LOG_LEVEL':'2','MPLBACKEND':'Agg'}
 report={'schema':3,'testedAt':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'device':'CPU','packages':packages,'chapters':[],'integrations':[],'optionalGPU':'not_run','learnerTrial':'not_run'}
 folders=[p for p in sorted(ROOT.iterdir()) if re.match(r'^\d{2}-',p.name) and (p/'tests').is_dir()]+[p for p in sorted(ROOT.glob('integration-project*')) if (p/'tests').is_dir()]
 out=ROOT/'visual-lab/course/runtime-v3.json'
 for p in folders:
  start=time.monotonic()
  try:
   run=subprocess.run([sys.executable,'-m','pytest','tests','-q','-rs','--disable-warnings'],cwd=p,env=env,text=True,capture_output=True,timeout=120)
   text=run.stdout+run.stderr;row={'directory':p.name,'sourceHash':digest(p),'exit':run.returncode,'seconds':round(time.monotonic()-start,2),'passed':0,'skipped':0}
   for k in ['passed','skipped','failed','errors']:
    found=re.findall(r'(\d+) '+k,text)
    if found:row[k]=int(found[-1])
   row['output']=text[-14000:];row['status']='pass' if run.returncode==0 and row['skipped']==0 else 'partial' if run.returncode in [0,5] else 'fail'
  except subprocess.TimeoutExpired:row={'directory':p.name,'sourceHash':digest(p),'exit':124,'status':'fail','output':'timeout120s'}
  report['integrations' if p.name.startswith('integration') else 'chapters'].append(row)
  out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  print(p.name,row['status'],row.get('passed',0),'passed',row.get('skipped',0),'skipped',flush=True)
 return int(any(r['exit'] not in [0,5] for r in report['chapters']+report['integrations']))
if __name__=='__main__':sys.exit(main())
