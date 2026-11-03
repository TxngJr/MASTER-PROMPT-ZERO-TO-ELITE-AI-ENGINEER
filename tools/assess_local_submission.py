#!/usr/bin/env python3
"""Test a learner src directory with an unchanged copy of chapter tests.
Runs learner Python locally. Not a sandbox or proof of authorship/understanding.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,tempfile
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
REFERENCE_DEPENDENCIES={80:['49-tokenizer-from-scratch/src/byte_bpe.py']}
def sha(data):return hashlib.sha256(data).hexdigest()
def folder_for(n):return next(ROOT.glob(f'{n:02d}-*'))
def suite_hash(n):
    folder=folder_for(n);paths=sorted((folder/'tests').rglob('*.py'))
    return sha(Path(__file__).read_bytes()+b''.join(str(p.relative_to(folder)).encode()+p.read_bytes()for p in paths)+b''.join(name.encode()+(ROOT/name).read_bytes()for name in REFERENCE_DEPENDENCIES.get(n,[])))
def source_hash(path):
    paths=sorted(p for p in path.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix not in ['.pyc','.pyo'])
    return sha(b''.join(str(p.relative_to(path)).encode()+b'\0'+p.read_bytes()+b'\0'for p in paths))
def identity(report):
    keys=['sourceSha256','suiteHash','seed','testedAt','sourceRole','contentVersion','chapter','suiteFamily','python','exitCode','passed','failed','skipped']
    return {**{k:report[k]for k in keys},'checks':[{'id':x['id'],'status':x['status']}for x in report['checks']]}
def assess(n,src,timeout=60,qa=False):
    folder=folder_for(n);expected=[p.relative_to(folder/'src')for p in (folder/'src').rglob('*.py')if '__pycache__'not in p.parts]
    if not expected:raise ValueError('No trusted source manifest for this chapter')
    if not src.is_dir()or any(not(src/p).is_file()for p in expected):raise ValueError('submission-src must contain the complete src tree; missing '+', '.join(str(p)for p in expected if not(src/p).is_file()))
    if src.resolve()==(folder/'src').resolve()and not qa:raise ValueError('Canonical course source is a reference, not a learner submission. Use a learner src directory; --qa is only for tool validation.')
    digest=source_hash(src);at=datetime.now(timezone.utc).isoformat();rows=[];logs='';exit_code=1
    with tempfile.TemporaryDirectory(prefix='course-learner-')as raw:
        workspace=Path(raw)/folder.name
        shutil.copytree(folder,workspace,ignore=shutil.ignore_patterns('src','__pycache__','.pytest_cache','animation.html'))
        shutil.copytree(src,workspace/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
        for name in REFERENCE_DEPENDENCIES.get(n,[]):
            destination=Path(raw)/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        xml=Path(raw)/'results.xml';env={**os.environ,'PYTHONPATH':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','MPLBACKEND':'Agg','CUDA_VISIBLE_DEVICES':''}
        try:
            proc=subprocess.run([sys.executable,'-m','pytest','tests','-q','-rs','--junitxml',str(xml)],cwd=workspace,env=env,text=True,capture_output=True,timeout=timeout)
            logs=(proc.stdout+proc.stderr)[-12000:];exit_code=proc.returncode
        except subprocess.TimeoutExpired:logs=f'Assessment exceeded {timeout}s';exit_code=124
        if xml.exists():
            for item in ET.parse(xml).iter('testcase'):
                status='skip'if item.find('skipped')is not None else'fail'if item.find('failure')is not None or item.find('error')is not None else'pass'
                rows.append({'id':item.attrib.get('classname','')+'::'+item.attrib.get('name',''),'status':status})
    changed=source_hash(src)!=digest
    if changed:
        rows=[{**r,'status':'fail'}for r in rows];exit_code=1;logs+='\nsource changed during assessment'
    passed=sum(r['status']=='pass'for r in rows);failed=sum(r['status']=='fail'for r in rows);skipped=sum(r['status']=='skip'for r in rows)
    if not rows or failed or skipped:exit_code=exit_code or 1
    content=json.loads((ROOT/'visual-lab/course/contracts.json').read_text())[str(n)]['contentVersion']
    report={'schemaVersion':1,'kind':'learner-code-run','suiteFamily':'chapter-required-tests','chapter':n,'contentVersion':content,'sourceRole':'qa-fixture'if qa else'learner-submission','sourcePath':str(src.resolve()),'sourceSha256':digest,'suiteHash':suite_hash(n),'seed':0,'testedAt':at,'python':platform.python_version(),'command':[sys.executable,'-m','pytest','tests','-q','-rs','--junitxml','results.xml'],'exitCode':exit_code,'passed':passed,'failed':failed,'skipped':skipped,'checks':rows,'learnerOutput':logs,'verification':'Unchanged chapter tests executed against a copy of this src hash. These tests do not cover every objective or prove authorship/understanding. Reviewer must inspect the source and new cases. JSON is not signed.'}
    report['referenceDependencies']=[{'path':name,'sha256':sha((ROOT/name).read_bytes()),'role':'course reference dependency; not learner-authored evidence'}for name in REFERENCE_DEPENDENCIES.get(n,[])]
    report['reportId']=sha(json.dumps(identity(report),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode());return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chapter',type=int,required=True);p.add_argument('--submission-src',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--qa',action='store_true');args=p.parse_args()
    if not 1<=args.chapter<=80:p.error('chapter must be 1..80')
    if args.output.resolve().is_relative_to(args.submission_src.resolve()):p.error('output must be outside submission-src so the report does not change the source hash')
    try:report=assess(args.chapter,args.submission_src.resolve(),qa=args.qa)
    except ValueError as exc:p.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'{report["passed"]} passed, {report["failed"]} failed, {report["skipped"]} skipped; sourceRole={report["sourceRole"]}')
    return 0 if report['exitCode']==0 else 1
if __name__=='__main__':sys.exit(main())
