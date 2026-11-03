#!/usr/bin/env python3
"""Acceptance evidence separated by structure/math/teaching/runtime/learner trials."""
import sys,json,subprocess,hashlib,re,argparse,os
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.visual_course_content import validate_contracts
from tools.verify_course_runtime import digest
from tools.assess_local_submission import source_hash as learner_source_hash,suite_hash as learner_suite_hash
LAB=ROOT/'visual-lab'
class Inspect(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.links=[];self.exercises=0;self.help=0
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id'in a:self.ids.append(a['id'])
  if tag in ['a','script','link','img']:
   link=a.get('href')or a.get('src')
   if link:self.links.append(link)
  if 'exercise-item'in a.get('class','').split():self.exercises+=1
  if 'help-panel'in a.get('class','').split():self.help+=1

def main():
 p=argparse.ArgumentParser();p.add_argument('--python-tests',action='store_true');p.add_argument('--runtime-python',default=str(ROOT/'.venv/frameworks/bin/python'));args=p.parse_args()
 if args.python_tests:subprocess.run([args.runtime_python,'tools/verify_course_runtime.py'],cwd=ROOT,check=True)
 contracts=json.loads((LAB/'course/contracts.json').read_text());validate_contracts(contracts)
 now=datetime.now(timezone.utc).isoformat();report={'schema':3,'checkedAt':now,'chapters':[],'batches':[],'learnerTrials':{'status':'not_run','reason':'ยังไม่มีผลการทดลองกับผู้เรียนจริง/ผลงานใหม่ที่ตรวจตาม rubric'},'GPU':{'status':'not_run','reason':'CPU checks do not validate CUDA, multi-GPU or deployment hardware'},'teachingReview':{'status':'authored-reference-review','scope':'1600 answers rewritten for the actual questions; explicit ID maps and objective-specific source references. Structural tests do not establish learning effectiveness.','unverified':['ผู้เรียนเข้าใจด้วยตนเองและถ่ายโอนความรู้ไปกรณีใหม่','ความครบถ้วนของคำอธิบายสำหรับผู้เรียนแต่ละพื้นฐาน','ความถูกต้องของผลงาน local ที่ผู้เรียนเขียนเพิ่มเติม','คำใบ้ที่สร้างจาก excerpt ยังต้องตรวจคุณภาพการชี้นำโดยผู้สอนรายข้อ']}}
 main_exercises=[e for c in contracts.values()for e in c['exercises']if e['number']]
 report['teachingReview']['hintSources']={kind:sum(e.get('hintSource')==kind for e in main_exercises)for kind in ['authored','gradual-solution-excerpt']}
 env={**os.environ,'VISUAL_LAB_FRAMEWORK_PYTHON':args.runtime_python,'OMP_NUM_THREADS':'1'}
 node=subprocess.run(['node','--test','visual-lab/tests/regression.test.mjs','visual-lab/tests/course-models.test.mjs','visual-lab/tests/quality-v3.test.mjs','visual-lab/tests/assessment-bank.test.mjs','visual-lab/tests/mastery.test.mjs'],cwd=ROOT,env=env,text=True,capture_output=True);report['numericalAndAssessmentTests']={'exit':node.returncode,'output':node.stdout+node.stderr,'testedAt':now}
 content=subprocess.run([args.runtime_python,'-m','pytest','visual-lab/tests/test_content_contracts.py','visual-lab/tests/test_learner_assessment.py','-q'],cwd=ROOT,text=True,capture_output=True);report['contractMutationTests']={'exit':content.returncode,'output':content.stdout+content.stderr,'testedAt':now}
 learner_qa=json.loads((LAB/'course/learner-runner-qa.json').read_text())if(LAB/'course/learner-runner-qa.json').exists()else{}
 report['learnerRunnerQA']={'testedAt':learner_qa.get('testedAt'),'sourceRole':'qa-fixture; no learner credit','passedChapters':learner_qa.get('passedChapters',0),'runnerFresh':learner_qa.get('runnerHash')==hashlib.sha256((ROOT/'tools/assess_local_submission.py').read_bytes()).hexdigest()}
 report['masteryPolicy']={'dimensions':6,'rounds':3,'retentionDays':7,'scope':'Only bounded criterion evidence reported by a reviewer; not 100% understanding, authenticated identity or signed attestation','actualLearnerEvaluations':'not_run'}
 runtime=json.loads((LAB/'course/runtime-v3.json').read_text())if(LAB/'course/runtime-v3.json').exists()else{};report['localRuntime']={k:v for k,v in runtime.items()if k not in ['chapters','integrations']};failures=[];quality={}
 for n in range(1,81):
  folder=next(ROOT.glob(f'{n:02d}-*'));page=folder/'animation.html';body=page.read_text();d=Inspect();d.feed(body);c=contracts[str(n)];issues=[]
  if d.exercises!=20:issues.append('expected20main exercises')
  if len(set(d.ids))!=len(d.ids):issues.append('duplicateHTML IDs')
  if re.search(r'__(?:[A-Z_]+)__',body):issues.append('unresolvedbuildmarker')
  if d.help<3:issues.append('missingtargetedhelp')
  if 'glossary'not in d.ids:issues.append('missingglossary')
  if 'mastery-review'not in d.ids:issues.append('missingcriterionassessmentpanel')
  runner_evidence=next((r for r in learner_qa.get('chapters',[])if r['chapter']==n),None)
  if not(report['learnerRunnerQA']['runnerFresh']and runner_evidence and runner_evidence.get('sourceRole')=='qa-fixture'and runner_evidence.get('suiteHash')==learner_suite_hash(n)and runner_evidence.get('sourceSha256')==learner_source_hash(folder/'src')and runner_evidence.get('exitCode')==0 and runner_evidence.get('skipped')==0):issues.append('learnerrunnerQAfailedorstale')
  for o in c['objectives']:
   if any(s not in d.ids for s in o['sections']):issues.append(o['id']+':unrenderedteaching')
   if o['id']not in d.ids:issues.append(o['id']+':missingproofrubric')
  for link in d.links:
   if re.match(r'^(https?:|mailto:|data:|#)',link):continue
   path=link.split('#')[0].split('?')[0]
   if path and not(folder/path).exists():issues.append('missinglocalasset:'+link)
  evidence=next((x for x in runtime.get('chapters',[])if x['directory']==folder.name),None)
  fresh=evidence and evidence.get('sourceHash')==digest(folder)
  corepass=bool(fresh and evidence['exit']==0 and evidence.get('skipped',0)==0)
  if not fresh:issues.append('runtimeevidencemissingorstale')
  if not corepass:issues.append('requiredlocaltestnotverified')
  row={'chapter':n,'contentVersion':c['contentVersion'],'pageHash':hashlib.sha256(page.read_bytes()).hexdigest(),'structure':'pass'if not issues else'fail','issues':issues,'objectives':len(c['objectives']),'authoredMainAnswers':20,'targetedHelp':d.help,'mathAndAssessment':'pass'if node.returncode==0 else'fail','teaching':'authored-reference-review; learner trial pending','localRuntime':evidence,'runtimeFresh':bool(fresh)};report['chapters'].append(row)
  if issues:failures.append(n)
  quality[str(n)]={'structure':row['structure'],'math':row['mathAndAssessment'],'teaching':'มีเฉลยเฉพาะข้อและ rubric; รอทดลองกับผู้เรียน','local':evidence if fresh else None,'testedAt':now,'runtimeAt':runtime.get('testedAt'),'contentVersion':c['contentVersion']}
 for b in range(1,28):
  rows=report['chapters'][(b-1)*3:b*3];directory='integration-project'if b==1 else f'integration-project-batch{b:02d}';folder=ROOT/directory;e=next((x for x in runtime.get('integrations',[])if x['directory']==directory),None);fresh=e and e.get('sourceHash')==digest(folder);passed=all(not r['issues']for r in rows)and node.returncode==0 and content.returncode==0 and fresh and e['exit']==0 and e.get('skipped',0)==0
  report['batches'].append({'batch':b,'chapters':[r['chapter']for r in rows],'engineeringChecks':'pass'if passed else'needs-review','teachingReview':'authored-reference-review','learnerTrials':'not_run','integration':e,'integrationFresh':bool(fresh)})
 report['summary']={'chapters':80,'mainAnswers':1600,'objectiveContracts':sum(len(c['objectives'])for c in contracts.values()),'engineeringFailures':failures,'localPassed':sum(x.get('passed',0)for x in runtime.get('chapters',[])),'integrationPassed':sum(x.get('passed',0)for x in runtime.get('integrations',[])),'skipped':sum(x.get('skipped',0)for x in runtime.get('chapters',[])+runtime.get('integrations',[]))}
 (LAB/'course/audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(LAB/'assets/quality-report.js').write_text('window.COURSE_QUALITY = '+json.dumps(quality,ensure_ascii=False,separators=(',',':'))+';\n')
 print(json.dumps(report['summary'],ensure_ascii=False));return int(bool(failures)or node.returncode!=0 or content.returncode!=0 or any(x['engineeringChecks']!='pass'for x in report['batches']))
if __name__=='__main__':sys.exit(main())
