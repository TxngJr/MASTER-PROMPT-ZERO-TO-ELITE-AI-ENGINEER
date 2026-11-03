#!/usr/bin/env python3
"""Compile authored IDs into contracts. A mapping file is the only matching authority."""
from pathlib import Path
import json,re,hashlib
from tools.visual_course_content import ROOT,source_parts,original_questions,authored_answers,validate_contracts
AUTH=ROOT/'visual-lab/course/authored'
def mappings(name):
 out={}
 for line in (AUTH/name).read_text().splitlines():
  if not line.strip()or line.startswith('#'):continue
  n,v=line.split(':',1);out[int(n)]=v.split()
 if name!='extra-objectives.txt' and set(out)!=set(range(1,81)):raise ValueError(name+': must map all 80 chapters')
 return out

def compile_contracts(chapters):
 om=mappings('objective-map.txt');extras=mappings('extra-objectives.txt');em=mappings('exercise-map.txt');answers=authored_answers();contracts={};authored_hints=json.loads((AUTH/'hints.json').read_text())
 if set(answers)!=set(range(1,81)):raise ValueError('Missing authored answers')
 for n in range(1,81):
  required={14:'dbscan',29:'multihead',60:'bootstrap'}.get(n)
  if required and required not in {lab['model']for lab in chapters[str(n)]['labs']}:raise ValueError(f'{n}: required mechanism {required} missing or misrouted')
  p=source_parts(n);folder=next(ROOT.glob(f'{n:02d}-*'));questions=original_questions(n)
  original=[]
  for part in p.values():
   if 'Learning Obj'in part['title']:original=re.findall(r'^\s*[-*]\s+(.+)',part['body'],re.M)
  entries=[]
  for i,raw in enumerate(om[n]+extras.get(n,[]),1):
   sections=[int(x)for x in raw.split(',')]
   if any(s not in p for s in sections):raise ValueError(f'{n}: bad objective source {raw}')
   entries.append({'id':f'ch{n:02d}-objective-{i:02d}','text':original[i-1]if i<=len(original) else ' / '.join(p[s]['title']for s in sections),'sections':[f'topic-{s}'for s in sections],'exerciseIds':[f'ch{n:02d}-objective-task-{i:02d}'],'evidenceIds':[f'ch{n:02d}-evidence-{i:02d}'],'labIds':[f'local-{n:02d}'],'sourceNumbers':sections})
  if original and len(om[n])!=len(original):raise ValueError(f'{n}: objective count mismatch')
  exercises=[]
  if len(em[n])!=20:raise ValueError(f'{n}: exercise map needs twenty entries')
  for k,raw in enumerate(em[n],1):
   sections=[int(x)for x in raw.split(',')]
   if any(s not in p for s in sections):raise ValueError(f'{n}/{k}: unknown source section')
   related=[o['id']for o in entries if any(f'topic-{s}'in o['sections']for s in sections)]
   # An unmapped exercise is an authoring error, never silently matched by text.
   if not related:raise ValueError(f'{n}/{k}: explicit source has no mapped objective: {raw}')
   answer=answers[n][k].replace('เปลี่ยn','เปลี่ยน').replace('judgew答案','judge answer')
   if re.match(r'(?i)^(define|give an operational|keep .* calculation|a complete answer|include one)',answer):raise ValueError(f'{n}/{k}: generic legacy answer')
   # Hints reveal the authored solution gradually; they never substitute another mechanism.
   # A short one-sentence solution must not be revealed in full by hint 2.
   # Derived hints are explicitly gradual excerpts, not claimed to be authored explanations.
   hints=authored_hints.get(str(n),{}).get(str(k))
   hint_source='authored' if hints else 'gradual-solution-excerpt'
   if not hints:hints=[f'ใช้แนวคิดในหัวข้อ {", ".join(p[s]["title"]for s in sections)}: ระบุสิ่งที่โจทย์ถามก่อนดูเฉลย','ส่วนต้นของแนวคำตอบ (ยังไม่ครบ): '+answer[:int(len(answer)*.4)]+'…','เปิดแนวคำตอบเพิ่ม แล้วลองเติมขั้นที่เหลือเอง: '+answer[:int(len(answer)*.72)]+'…']
   if len(hints)!=3 or len(set(hints))!=3 or answer in hints:raise ValueError(f'{n}/{k}: hints must be three distinct stages, separate from full solution')
   exercises.append({'id':f'ch{n:02d}-exercise-{k:02d}','number':k,'question':questions[k],'answer':answer,'hints':hints,'hintSource':hint_source,'sections':[f'topic-{s}'for s in sections],'objectiveIds':related,'answerTopicIds':[f'topic-{s}'for s in sections]})
  evidence=[]
  for o in entries:
   reference='\n\n'.join(p[num]['body']for num in o['sourceNumbers'])
   examples=[x for x in exercises if set(x['sections']).intersection(o['sections'])]
   worked=examples[0]['answer'] if examples else reference
   question=f"{o['text']}: ใช้กรณีในหัวข้อ {', '.join(p[num]['title']for num in o['sourceNumbers'])} เป็น reference แล้วสร้างกรณีใหม่ด้วยตัวเอง แสดง input/หน่วย/shape หรือ state ก่อนทำงาน ไล่ operation จนได้ผล พร้อมกรณีที่กลไกนี้ใช้ไม่ได้และหลักฐานตรวจคำตอบ"
   task={'id':o['exerciseIds'][0],'number':None,'question':question,'answer':reference,'hints':['เริ่มจากนิยามและสมมติฐานใน reference ที่เชื่อมตรงกับ objective นี้',worked,reference], 'sections':o['sections'],'objectiveIds':[o['id']],'answerTopicIds':o['sections'],'kind':'objective-proof'}
   exercises.append(task)
   evidence.append({'id':o['evidenceIds'][0],'objectiveId':o['id'],'type':'learner-work','task':question,'sections':o['sections'],'exerciseIds':o['exerciseIds'],'reference':reference,'workedAnswer':worked,'requiredArtifacts':['คำอธิบายกลไกตามนิยามและสมมติฐานใน reference นี้','กรณีใหม่พร้อมค่าระหว่างทางและผลที่ตรวจด้วย reference implementation หรือการคำนวณมือ','กรณีล้มเหลวของกลไกนี้ พร้อมสาเหตุและวิธีตรวจ'], 'passingExample':worked,'failingExample':'ตอบเพียงชื่อหัวข้อหรือผลสุดท้าย โดยข้ามนิยาม/เงื่อนไข/operation ที่ reference นี้กำหนด','verification':'requires learner submission and rubric review; not automatically passed by reading'})
  c={'chapter':n,'contentVersion':'3.0.0','objectives':entries,'exercises':exercises,'evidence':evidence,'sectionIds':[f'topic-{s}'for s in p],'localLab':{'id':f'local-{n:02d}','sourceFiles':[str(x.relative_to(ROOT))for x in sorted((folder/'src').glob('*.py'))],'testFiles':[str(x.relative_to(ROOT))for x in sorted((folder/'tests').glob('test_*.py'))]}}
  payload=json.dumps(c,sort_keys=True,ensure_ascii=False).encode();c['contentVersion']='3.0.0+'+hashlib.sha256(payload).hexdigest()[:12];contracts[str(n)]=c
 validate_contracts(contracts)
 return contracts
if __name__=='__main__':
 c=compile_contracts(json.loads((ROOT/'visual-lab/course/chapters.json').read_text()));(ROOT/'visual-lab/course/contracts.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');print('Validated',sum(len(x['objectives'])for x in c.values()),'objectives and 1600 authored answers')
