"""Content contract helpers. No heuristic matching of exercises to topics."""
from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def numbered(text):
    result={};cur=None;lines=[];fenced=False
    for line in text.splitlines():
        if line.strip().startswith(('```','~~~')):fenced=not fenced
        m=re.match(r'^\s*(\d+)[.)]\s+(.*)',line)if not fenced else None
        if m:
            if cur is not None:result[cur]='\n'.join(lines).strip()
            cur=int(m[1]);lines=[m[2]]
        elif line.startswith('#')and not fenced:
            if cur is not None:result[cur]='\n'.join(lines).strip()
            cur=None;lines=[]
        elif cur is not None:lines.append(line)
    if cur is not None:result[cur]='\n'.join(lines).strip()
    return result

def original_questions(n):
    folder=next(ROOT.glob(f'{n:02d}-*'));q=numbered((folder/'exercises/README.md').read_text())
    if len(q)<20:
        text=(ROOT/'SUPPLEMENTARY_EXERCISE_WORKBOOK.md').read_text();part=re.search(r'^## '+re.escape(folder.name)+r' —.*?(?=^## |\Z)',text,re.M|re.S)
        if part:
            for k,v in numbered(part.group()).items():q.setdefault(k,v)
    if sorted(q)!=list(range(1,21)):raise ValueError(f'chapter {n} exercises must be 1–20')
    return q

def source_parts(n):
    folder=next(ROOT.glob(f'{n:02d}-*'));text=(folder/'README.md').read_text();out={}
    for block in re.split(r'(?m)(?=^## )',text):
        m=re.match(r'^## (\d+)\. ([^\n]+)',block)
        if m:out[int(m[1])]={'title':m[2],'body':block}
    return out

def authored_answers():
    result={}
    for p in sorted((ROOT/'visual-lab/course/authored').glob('answers-*.md')):
        for block in re.split(r'(?m)(?=^## \d{2}\s*$)',p.read_text()):
            m=re.match(r'^## (\d{2})\s*\n',block)
            if not m:continue
            n=int(m[1]);answers=numbered(block[m.end():])
            if sorted(answers)!=list(range(1,21)):raise ValueError(f'{p}: chapter {n} needs 20 authored answers; got {sorted(answers)}')
            if n in result:raise ValueError(f'duplicate chapter answers {n}')
            result[n]=answers
    # Regression's detailed 20 worked solutions are the verified reference.
    if 6 not in result:result[6]=numbered((ROOT/'06-regression/solutions/README.md').read_text())
    return result

def validate_contracts(contracts):
    errors=[];ids=set()
    if set(contracts)!=set(map(str,range(1,81))):errors.append('expected exactly chapters 1–80')
    for key,c in contracts.items():
        for kind in ['objectives','exercises','evidence']:
            for item in c[kind]:
                if item['id']in ids:errors.append('duplicate id '+item['id'])
                ids.add(item['id'])
        ex={e['id']:e for e in c['exercises']};ev={e['id']:e for e in c['evidence']}
        if len(ex)<20:errors.append(f'{key}: fewer than 20 exercises')
        for o in c['objectives']:
            if not o['sections']or not o['exerciseIds']or not o['evidenceIds']:errors.append(o['id']+': incomplete coverage')
            for eid in o['exerciseIds']:
                if eid not in ex:errors.append(o['id']+': missing exercise '+eid)
            for eid in o['evidenceIds']:
                if eid not in ev:errors.append(o['id']+': missing evidence '+eid)
        sections=set(c.get('sectionIds',[]))
        primary=[e for e in ex.values()if e.get('number')is not None]
        if sorted(e['number']for e in primary)!=list(range(1,21)):errors.append(key+': main exercises must be exactly1–20')
        for o in c['objectives']:
            if not set(o['sections'])<=sections:errors.append(o['id']+': unknown teaching section')
            if o.get('labIds')!=[c['localLab']['id']]:errors.append(o['id']+': invalid lab reference')
            for eid in o['evidenceIds']:
                if eid in ev and ev[eid].get('objectiveId')!=o['id']:errors.append(o['id']+': evidence belongs to a different objective')
        for e in ex.values():
            if len(e.get('answer','').strip())<35:errors.append(e['id']+': answer too short')
            if len(e.get('hints',[]))!=3 or any(not h.strip()for h in e.get('hints',[])):errors.append(e['id']+': need three hints')
            if not e.get('objectiveIds'):errors.append(e['id']+': no objective mapping')
            if not e.get('sections')or not set(e['sections'])<=sections:errors.append(e['id']+': no valid explicit teaching reference')
            if e.get('answerTopicIds')!=e.get('sections'):errors.append(e['id']+': answer belongs to different topic')
            if re.search(r'(?i)^(define .+purpose|keep the .+calculation visible|a complete answer includes|ข้อนี้ตรวจด้วยผลงานและเหตุผลตามเกณฑ์)',e.get('answer','')):errors.append(e['id']+': generic rubric is not an answer')
    if errors:raise ValueError('\n'.join(errors))
    return True
