#!/usr/bin/env python3
"""Validate copied-src assessment workspaces for all chapters; never learner credit."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.assess_local_submission import assess,folder_for

def main():
 rows=[]
 with ThreadPoolExecutor(max_workers=4)as executor:
  futures={executor.submit(assess,n,folder_for(n)/'src',qa=True):n for n in range(1,81)}
  for future in as_completed(futures):
   r=future.result();rows.append(r)
   if r['exitCode']!=0:print('FAIL',r['chapter'],r['learnerOutput'][-2200:],flush=True)
   elif len(rows)%10==0:print('QA copied-src chapters checked:',len(rows),flush=True)
 rows.sort(key=lambda r:r['chapter'])
 report={'schemaVersion':1,'testedAt':datetime.now(timezone.utc).isoformat(),'sourceRole':'qa-fixture','runnerHash':hashlib.sha256((ROOT/'tools/assess_local_submission.py').read_bytes()).hexdigest(),'scope':'Copied canonical src validates learner-runner plumbing only; no learner credit','chapters':rows,'passedChapters':sum(r['exitCode']==0 for r in rows),'failedChapters':[r['chapter']for r in rows if r['exitCode']!=0]}
 (ROOT/'visual-lab/course/learner-runner-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('Copied-src QA:',report['passedChapters'],'/80; no learner credit');return bool(report['failedChapters'])
if __name__=='__main__':sys.exit(main())
