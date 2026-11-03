#!/usr/bin/env python3
"""Build static, offline lessons from authored, versioned content contracts."""
from pathlib import Path
import argparse,html,json,re,subprocess,sys,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import markdown
from tools.visual_course_content import ROOT,source_parts
from tools.build_visual_contracts import compile_contracts,mappings
from tools.visual_course_shapes import case as shape_case
from tools.assess_regression import suite_hash as regression_suite_hash
from tools.assess_local_submission import suite_hash as chapter_suite_hash
LAB=ROOT/'visual-lab'
def esc(x):return html.escape(str(x),quote=True)
ids=itertools.count()
def md(text,base=''):
 text=re.sub(r'(?is)<script\b.*?</script>','',text)
 s=markdown.markdown(text,extensions=['fenced_code','tables','sane_lists'])
 s=re.sub(r'<(/?)h([1-5])',lambda m:'<'+m[1]+'h'+str(int(m[2])+1),s)
 s=s.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
 if base:
  def fix(m):
   x=m[1]
   if re.match(r'^(https?:|mailto:|#|data:)',x):return m[0]
   return 'href="../'+str(Path(base)/x)+'"'
  s=re.sub(r'href="([^"]+)"',fix,s)
 return s

def helper(key,answer,sample=None):
 nodes=sample['result']['nodes']if sample else[]
 formula='<ol>'+''.join('<li><code>'+esc(op['formula'])+'</code> = '+esc(json.dumps(op['value'],ensure_ascii=False))+'<p>'+esc(op['explanation'])+'</p></li>'for op in nodes)+'</ol>'if nodes else md(answer)
 panels=[('terms','ศัพท์ / สัญลักษณ์','<p>เปิดอภิธานศัพท์เพื่ออ่านคำไทยและหน่วย แล้วอ่านคำอธิบายที่ตรงกับกรณีนี้:</p>'+md(answer)+'<a href="#glossary">อภิธานศัพท์</a>'),('equation','สมการ',formula),('visual','ภาพและ shape',('<p>'+esc(sample['result']['note'])+'</p>'+formula if sample else md(answer))),('code','โค้ด',md(answer)+'<p><a href="#implementation">ดู implementation และ tests ของบท</a> · <a href="#local-lab">วิธีรัน local lab</a></p>'),('result','ผลการทดลอง',formula+'<p>ตรึง baseline ก่อนเปลี่ยนข้อมูล ใช้ค่าใน trace นี้ตรวจขั้นแรกที่ต่างจากการทำนาย</p>')]
 return '<details class="help-panel"><summary>ยังไม่เข้าใจตรงนี้</summary><p>เลือกจุดที่ติด เพื่อเปิดคำอธิบายและตัวอย่างของหัวข้อนี้ (ใช้ได้ offline)</p>'+''.join(f'<details id="help-{esc(key)}-{kind}"><summary>{title}</summary>{body}</details>'for kind,title,body in panels)+'</details>'

def choice_variant(vid,question,context,correct,wrong,explanation,section):
 return dict(id=vid,type='choice',question=question,context=context,options=[{'id':'correct','text':str(correct)},*[{'id':f'error-{i}','text':str(x)}for i,x in enumerate(wrong)]],answer='correct',explanation=explanation,feedback='ขั้นแรกที่ต้องแก้: '+explanation,remediation={'section':section,'kind':'equation'})

def checks_for(n,ch,contract,samples,old):
 primary=ch['sections'][0]['slug'];oid=contract['objectives'][0]['id'];out=[]
 def add(cat,variants,objective_id=None):out.append(dict(id=f'ch{n:02d}-{cat}',category=cat,objectiveId=objective_id or oid,version=contract['contentVersion'],variants=variants))
 q=old['quiz'];add('principle',[choice_variant('concept-1',q['question'],'พิจารณากลไกและข้อจำกัด',q['answer'],q['options']if 'options'in q else [q['distractor']]if q.get('distractor')else[x for x in q.get('choices',[])if x!=q['answer']],q['explanation'],primary)])
 # Legacy quiz shape is normalized explicitly, never parsed from answer prose.
 if len(out[0]['variants'][0]['options'])<2:
  out[0]['variants'][0]['options'].append({'id':'error-guarantee','text':'ผลตัวอย่างเดียวรับประกันผลทุก input และทุกกรณีขอบเขต'})
 for cat,offset in [('calculate',0),('predict',1),('transfer',2)]:
  variants=[]
  lab=ch['labs'][min(offset,len(ch['labs'])-1)]
  teaching=next((s for s in ch['sections']if s['lab']==lab['id']),None)
  section=teaching['slug']if teaching else primary
  related=next((e['objectiveIds']for e in contract['exercises']if teaching and e['id']==teaching['exerciseId']),[oid])
  for v in range(3):
   sample=samples[ch['labs'][min(offset,len(ch['labs'])-1)]['id']];pool=[{'state':sample['state'],'result':sample['result']},*sample['variants']]
   item=pool[min(v+offset,len(pool)-1)];r=item['result'];metric=next((m for m in r['metrics']if isinstance(m[1],(int,float))and not isinstance(m[1],bool)),None)
   if metric is None:raise ValueError(f'{n}: no numerical metric for {cat}')
   trace=' → '.join(f'{op["name"]}: {op["formula"]} = {json.dumps(op["value"],ensure_ascii=False)}'for op in r['nodes'])
   title={'calculate':'คำนวณจาก input ที่กำหนด','predict':'ทำนายผลของการตั้งค่าใหม่','transfer':'ประยุกต์กับกรณีใหม่'}[cat]
   variants.append(dict(id=f'{cat}-{v}',type='number',question=f'{title} ใน {ch["labs"][min(offset,len(ch["labs"])-1)]["title"]}: {metric[0]} เท่าไร?',context=r['note'],initial=item['state'],trace=[{'id':f'op-{idx}','name':op['name'],'formula':op['formula'],'answer':op['value']}for idx,op in enumerate(r['nodes'])if isinstance(op['value'],(int,float))or isinstance(op['value'],list)and len(json.dumps(op['value']))<180],answer=metric[1],explanation=f'{metric[0]}={metric[1]}. '+trace,feedback='เริ่มตรวจ input และ reduction ของ metric นี้ก่อน: '+trace,remediation={'section':section,'kind':'result'}))
  unique=[];seen=set()
  for variant in variants:
   signature=json.dumps(variant.get('initial',{}),sort_keys=True)
   if signature not in seen:unique.append(variant);seen.add(signature)
  add(cat,unique,related[0])
 add('shape-state',[choice_variant(f'shape-{v}',shape_case(n,v)[1],shape_case(n,v)[0],shape_case(n,v)[2],shape_case(n,v)[3],shape_case(n,v)[4],primary)for v in range(3)])
 add('debug',[choice_variant('debug-1','ข้อความใดอธิบายข้อผิดพลาด/ข้อจำกัดของบทนี้ได้ถูกต้อง?',old['title'],old['misconception'],[old['quiz']['distractor']],old['misconception'],primary)])
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--chapter',type=int);args=parser.parse_args()
 chapters=json.loads((LAB/'course/chapters.json').read_text());contracts=compile_contracts(chapters);sectionmap=mappings('section-exercises.txt')
 policy=json.loads((LAB/'course/authored/mastery-policy.json').read_text());runtime=json.loads((LAB/'course/runtime-v3.json').read_text())
 js="""const fs=require('fs'),vm=require('vm'),c={window:{}};for(const f of ['lessons','chapter-meta'])vm.runInNewContext(fs.readFileSync('visual-lab/assets/'+f+'.js','utf8'),c);process.stdout.write(JSON.stringify({lessons:c.window.VISUAL_LESSONS,meta:c.window.CHAPTER_META}));"""
 legacy=json.loads(subprocess.check_output(['node','-e',js],cwd=ROOT,text=True))
 compute="""const fs=require('fs'),vm=require('vm'),c={};for(const f of ['course-math','course-models','course-models-v3'])vm.runInNewContext(fs.readFileSync('visual-lab/assets/'+f+'.js','utf8'),c);const chapters=JSON.parse(fs.readFileSync('visual-lab/course/chapters.json','utf8')),out={};for(const[n,ch]of Object.entries(chapters)){out[n]={};for(const l of ch.labs){const m=c.CourseMath.models[l.model],state={...m.defaults,...l.initial},result=m.compute(state),variants=[];for(const control of m.controls.filter(v=>v.type==='range'))for(const value of [control.min,control.max]){if(value===state[control.key])continue;try{const next={...state,[control.key]:value},changed=m.compute(next);variants.push({state:next,result:changed,key:control.key,value})}catch{}}out[n][l.id]={state,result,controls:m.controls,variants};}}process.stdout.write(JSON.stringify(out));"""
 samples=json.loads(subprocess.check_output(['node','-e',compute],cwd=ROOT,text=True));bookdata={}
 for key,ch in chapters.items():
  n=int(key);ch['id']=n;c=contracts[key];directory=legacy['meta'][key]['directory'];folder=ROOT/directory;parts=source_parts(n);old=next(x for x in legacy['lessons']if x['id']==n)
  import hashlib
  buildhash=hashlib.sha256(b''.join((ROOT/path).read_bytes()for path in ['tools/build_visual_book.py','tools/visual_course_shapes.py','visual-lab/assets/course-math.js','visual-lab/assets/course-models.js','visual-lab/assets/course-models-v3.js','visual-lab/assets/course-assessment.js','visual-lab/assets/course-app.js','visual-lab/assets/course-render.js','visual-lab/assets/course.css','visual-lab/template.html','visual-lab/course/authored/regression-teaching.json','visual-lab/course/authored/glossary.json'])).hexdigest()[:12]
  buildhash=hashlib.sha256((buildhash+''.join(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()for path in ['tools/assess_regression.py','tools/assess_local_submission.py','visual-lab/assets/course-mastery.js','visual-lab/assets/course-mastery-panel.js','visual-lab/course/authored/mastery-policy.json','visual-lab/assessment/regression/starter.py'])).encode()).hexdigest()[:12]
  c['contentVersion']+='.'+buildhash
  ch['masteryPolicy']={k:v for k,v in policy.items()if k!='regressionChecks'}
  for o in c['objectives']:o['assessmentDimensions']=[d['id']for d in policy['dimensions']];o['assessmentRounds']=[r['id']for r in policy['rounds']]
  local_runtime=next(x for x in runtime['chapters']if x['directory']==directory)
  ch['runtimeAssessment']={'family':'regression-challenge'if n==6 else'chapter-required-tests','suiteHash':regression_suite_hash()if n==6 else chapter_suite_hash(n),'checkCount':len(policy['regressionChecks'])if n==6 else local_runtime['passed'],**({'checkIds':list(policy['regressionChecks'])}if n==6 else{})}
  c['masteryPolicy']=ch['masteryPolicy'];c['runtimeAssessment']=ch['runtimeAssessment']
  ch['contentVersion']=c['contentVersion'];ch['objectiveContracts']=c['objectives'];ch['objectives']=[o['text']for o in c['objectives']];ch['sourceObjectives']=ch['objectives'];ch['evidence']=c['evidence']
  ch['completionCriteria']=['อ่านหัวข้อและอธิบายกลไกด้วยคำของตัวเอง','ผ่านคำถามฝึกในเว็บและโจทย์ตัวเลขกรณีใหม่แยกกัน','ทำแบบฝึก 20 ข้อ พร้อมตรวจด้วยเฉลยเฉพาะข้อ','ส่งหลักฐานราย objective และผลงาน local ตาม rubric; เว็บไม่ได้รัน Python แทนคุณ']
  article='';ch['sourceSections']=[]
  if len(sectionmap[n])!=len(ch['sections']):raise ValueError(f'{n}: missing authored teaching-section links')
  for i,s in enumerate(ch['sections']):
   ex=dict(c['exercises'][int(sectionmap[n][i])-1]);
   if n==6:ex['answer']=json.loads((LAB/'course/authored/regression-teaching.json').read_text())[s['slug']]
   sample=samples[key][s['lab']];s['intuition']=ex['answer'];s['symbols']='; '.join(f'{x["key"]}: {x["label"]},initial={sample["state"][x["key"]]}'for x in sample['controls']);s['derivation']=[f'{op["formula"]} = {json.dumps(op["value"],ensure_ascii=False)}; {op["explanation"]}'for op in sample['result']['nodes']];failure=c['exercises'][(16 if n<=12 or n>=79 else 14 if n<=33 else 15)-1]['answer'];s['failure']=failure+'; '+sample['result']['note'];s['exerciseId']=ex['id']
   article+=f'<section id="{esc(s["slug"])}" class="teaching-section"><span class="section-kicker">CHAPTER {n:02d} · ภาพ → สมการ → โค้ด</span><h2>{esc(s["title"])}</h2><h3>โจทย์และเหตุผล</h3><p>{esc(s["why"])}</p><h3>ตัวอย่างและคำอธิบาย</h3>{md(ex["question"])}{md(ex["answer"])}<div class="symbols"><h3>สัญลักษณ์ / input controls</h3><p>{esc(s["symbols"])}</p></div><h3>แทนค่าจริงทีละ operation</h3><ol>'+''.join('<li>'+esc(x)+'</li>'for x in s['derivation'])+f'</ol><p>{esc(ch["labs"][next(j for j,l in enumerate(ch["labs"])if l["id"]==s["lab"])]["question"])}</p><a href="#workbench" data-open-lab="{esc(s["lab"])}">เปิดการทดลองนี้</a><p class="failure-box">{esc(s["failure"])}</p>'+helper(s['slug'],ex['answer'],sample)+f'<label class="read-marker"><input type="checkbox" data-read="{esc(s["slug"])}"> อ่านและอธิบายหัวข้อนี้แล้ว</label></section>'
  canonical=(folder/'README.md').read_text();canonical=re.sub(r'## Interactive Animation Lab.*?(?=\n## \d)','',canonical,flags=re.S)
  addenda=(ROOT/'CHAPTER_COMPLETION_ADDENDA.md').read_text();companion=re.search(r'^## '+re.escape(directory)+r' —.*?(?=^## |\Z)',addenda,re.M|re.S)
  if companion:canonical+='\n\n'+companion.group()
  aliashtml=markdown.markdown(canonical,extensions=['fenced_code','tables','toc']);aliases={num:[]for num in parts};current=next(iter(parts));prefix='source-'+re.sub(r'[^a-zA-Z0-9]+','-',directory)+'-canonical-'
  for level,aid,heading in re.findall(r'<h([1-6]) id="([^"]+)">(.*?)</h[1-6]>',aliashtml,re.S):
   match=re.match(r'(\d+)\.',re.sub('<[^>]+>','',heading))
   if level=='2'and match and int(match[1])in parts:current=int(match[1])
   aliases[current].append(prefix+aid)
  article+='<section id="reference" class="canonical-section"><h2>เนื้อหาครบทุกหัวข้อ พร้อมแผนที่ objective</h2><nav class="source-toc">'+''.join(f'<a href="#chapter-{n}/section/topic-{num}">{num}. {esc(p["title"])}</a> 'for num,p in parts.items())+'</nav>'
  for num,p in parts.items():
   slug=f'topic-{num}';ch['sourceSections'].append({'slug':slug,'title':p['title']});related=[e for e in c['exercises']if e['number']and slug in e['sections']]
   article+=f'<section id="{slug}" class="source-topic">'+''.join('<span id="'+esc(alias)+'" class="legacy-anchor"></span>'for alias in aliases[num])+md(p['body'],directory)
   if related:article+='<h3>อธิบายภาษาไทยและตัวอย่างที่ใช้ตรวจความเข้าใจ</h3>'+''.join(md(e['answer'])for e in related[:2])+helper(slug,related[0]['answer'])
   if not related:article+=helper(slug,p['body'])
   article+=f'<label class="read-marker"><input type="checkbox" data-read="{slug}"> อ่านหัวข้อนี้แล้ว</label></section>'
  if companion:article+='<details><summary>รายละเอียดเพิ่มเติมและข้อกำหนดของ local lab</summary>'+md(companion.group(),directory)+'</details>'
  article+='</section><section id="objective-evidence" class="canonical-section"><h2>พิสูจน์ความสามารถทีละ objective</h2><p>ทำกรณีใหม่ก่อนเปิด reference แล้วบันทึกงานด้านล่าง ให้ผู้ตรวจใช้ rubric ตรวจคำอธิบาย ผลคำนวณ โค้ด และข้อจำกัด บันทึกเก็บในเบราว์เซอร์เครื่องนี้ กดส่งออกเพื่อสำรองหรือส่งให้ผู้ตรวจ เว็บไม่ได้รันโค้ดหรือตัดสินผลงานแทนผู้ตรวจ</p><button type="button" class="secondary-button" data-export-evidence>ส่งออกหลักฐานบทนี้เป็น JSON</button><p role="status" id="evidence-export-status"></p>'
  for o,ev in zip(c['objectives'],c['evidence']):
   article+=f'<details id="{o["id"]}" class="objective-proof"><summary>{esc(o["text"])}</summary><p>{esc(ev["task"])}</p><div class="evidence-notebook" data-evidence="{ev["id"]}"><h3>งานของฉันสำหรับ objective นี้</h3>'
   for field,label,prompt in [('explanation','อธิบายด้วยคำของตัวเอง','ระบุนิยาม สมมติฐาน และเหตุผลของ operation ที่ใช้'),('calculation','กรณีใหม่และการคำนวณ','กำหนด input หน่วย shape/state และค่าระหว่างทาง ตรวจคำตอบด้วยวิธีอิสระ'),('implementation','โค้ดและหลักฐานทดสอบบนเครื่อง','บันทึกโค้ดหรือ path, คำสั่งรัน, environment และผลที่รันจริง หากยังไม่ได้รันให้ระบุไว้'),('limits','กรณีล้มเหลวและวิธีแก้','แสดง input ที่ทำให้ล้มเหลว อาการ ขั้นแรกที่ผิด และวิธีตรวจ/แก้')]:
    article+=f'<label for="{ev["id"]}-{field}">{label}</label><textarea id="{ev["id"]}-{field}" data-evidence-text="{field}" rows="4" placeholder="{prompt}"></textarea>'
   article+='<fieldset><legend>ฉันตรวจงานตนเองตามรายการนี้แล้ว</legend>'+''.join(f'<label><input type="checkbox" data-evidence-review="{i}"> {esc(x)}</label>'for i,x in enumerate(ev['requiredArtifacts']))+'</fieldset><p class="evidence-status" role="status"></p></div><details><summary>ตัวอย่างคำตอบที่ใช้เป็นจุดเริ่มต้น + reference เฉพาะเรื่อง</summary>'+md(ev['workedAnswer'])+md(ev['reference'])+'</details><p>ไม่ผ่าน: '+esc(ev['failingExample'])+'</p></details>'
  article+='</section><section id="implementation" class="canonical-section"><h2>Implementation และ tests ที่รันบนเครื่องได้</h2>'
  for kind in ['sourceFiles','testFiles']:
   for path in c['localLab'][kind]:article+=f'<details><summary>{esc(path)}</summary><pre><code>{esc((ROOT/path).read_text())}</code></pre></details>'
  article+='</section><section id="glossary" class="canonical-section"><h2>อภิธานศัพท์ — คำไทย ความหมาย และข้อควรตรวจ</h2><dl>'+''.join('<dt>'+esc(term)+'</dt><dd>'+esc(body)+'</dd>'for term,body in json.loads((LAB/'course/authored/glossary.json').read_text()).items())+'</dl></section>'
  exhtml='<h2>แบบฝึก 20 ข้อ — เฉลยเฉพาะข้อ</h2><p>ทำคำตอบก่อนเปิด hints; โจทย์ปลายเปิดให้ใช้ตัวอย่างคำตอบและหลักฐานราย objective ตรวจงานจริง</p>'
  for e in c['exercises']:
   if not e['number']:continue
   exhtml+=f'<section id="exercise-{e["number"]}" class="exercise-item"><h3>ข้อ {e["number"]}</h3>{md(e["question"])}'+''.join(f'<details><summary>คำใบ้ขั้นที่ {i+1}</summary>{md(h)}</details>'for i,h in enumerate(e['hints']))+'<details><summary>เฉลยพร้อมเหตุผล / เกณฑ์ตรวจเฉพาะข้อนี้</summary>'+md(e['answer'])+'<p>อ้างอิง: '+', '.join(f'<a href="#chapter-{n}/section/{slug}">{esc(parts[int(slug.split("-")[1])]["title"])}</a>'for slug in e['sections'])+f'</p></details><label class="read-marker"><input type="checkbox" data-exercise-reviewed="{e["id"]}"> ฉันทำข้อนี้และตรวจตามเฉลยด้วยตนเองแล้ว</label></section>'
  batch=(n-1)//3+1;integration='integration-project'if batch==1 else f'integration-project-batch{batch:02d}'
  local=f'<section id="local-lab"><h2>Local lab: Python 3.12 บน CPU</h2><p>รันจาก repository root; ใช้ Python 3.12 แยก environment สำหรับ framework ตาม requirements ของชุดบท</p><pre><code>python3.12 -m venv .venv-course\nsource .venv-course/bin/activate\npython -m pip install -r requirements-batch{batch:02d}.txt pytest\n(cd {directory} &amp;&amp; python -m pytest tests -q -rs)</code></pre><p>ผลที่คาดหวัง: required tests ผ่านและไม่มี required test ถูก skip; GPU/integrations ที่ยังไม่รันต้องระบุแยก Python 3.15 ที่มากับเครื่องอาจใช้กับ framework wheels เหล่านี้ไม่ได้</p>'+md((folder/'mini-project/README.md').read_text(),directory+'/mini-project')+md((ROOT/integration/'README.md').read_text(),integration)+'</section>'
  if n==6:local+='<pre><code>python visual-lab/python/regression_walkthrough.py\nnode --test visual-lab/tests/regression.test.mjs</code></pre><p>expected MSE3,gradw−4,gradb−2; updateη.1 → w1.4,b.2,MSE1.36</p>'
  coverage='<p>ผูกด้วย ID ที่ระบุใน source contract โดยตรง; หลักฐานผลงานต้องตรวจด้วย rubric และ tests ของกรณีใหม่</p><div class="table-scroll"><table><thead><tr><th>Objective ID / ความสามารถ</th><th>คำสอน</th><th>Lab</th><th>โจทย์ → หลักฐาน</th></tr></thead><tbody>'
  for o in c['objectives']:coverage+=f'<tr><td>{esc(o["id"])}<br>{esc(o["text"])}</td><td>'+', '.join(f'<a href="#chapter-{n}/section/{s}">{s}</a>'for s in o['sections'])+f'</td><td>{esc(c["localLab"]["id"])}</td><td><a href="#chapter-{n}/section/{o["id"]}">{esc(o["exerciseIds"][0])} → {esc(o["evidenceIds"][0])}</a></td></tr>'
  coverage+='</tbody></table></div>';ch['checks']=checks_for(n,ch,c,samples[key],old);bookdata[key]=dict(article=article,exercises=exhtml,local=local,coverage=coverage)
 (LAB/'course/contracts.json').write_text(json.dumps(contracts,ensure_ascii=False,indent=2)+'\n')
 (LAB/'course/lab-fixtures.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
 for filename,var,data in [('course-catalog','COURSE_CATALOG',chapters),('book-content','BOOK_DATA',bookdata)]: (LAB/f'assets/{filename}.js').write_text(f'window.{var} = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
 template=(LAB/'template.html').read_text()
 for key,ch in chapters.items():
  if args.chapter and int(key)!=args.chapter:continue
  b=bookdata[key]
  def make(assets):
   result=template.replace('__TITLE__',esc(ch['title'])).replace('__CHAPTER__',key).replace('__ASSETS__',assets).replace('__ROOT__','../')
   for name in ['ARTICLE','EXERCISES','LOCAL_LAB']:result=result.replace('__'+name+'__',b[{'ARTICLE':'article','EXERCISES':'exercises','LOCAL_LAB':'local'}[name]])
   return result
  (ROOT/legacy['meta'][key]['directory']/'animation.html').write_text(make('../visual-lab/assets'))
  if int(key)==1 and not args.chapter or args.chapter==6:(LAB/'index.html').write_text(make('assets'))
 print('Compiled 80 chapters, 1600 authored answers, '+str(sum(len(c['objectives'])for c in contracts.values()))+' objective evidence contracts; content v3')
if __name__=='__main__':main()
