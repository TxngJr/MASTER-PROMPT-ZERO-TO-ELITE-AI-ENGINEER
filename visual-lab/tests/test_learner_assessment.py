"""Meaningful positive/negative checks: never count QA as a learner's result."""
from pathlib import Path
import importlib.util,json,sys,subprocess,hashlib,types
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.assess_regression import run_checks,CHECK_IDS,identity,sha
spec=importlib.util.spec_from_file_location('qa_adapter',ROOT/'visual-lab/tests/reference/learner_regression_qa.py')
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
@pytest.mark.parametrize('seed',[0,7,20261015,2**32-1])
def test_complete_adapter_passes_fresh_independent_cases(seed):
    rows=run_checks(qa,seed);assert len(rows)==15
    assert all(r['status']=='pass'for r in rows),rows
@pytest.mark.parametrize('defect,check',[
 ('broadcast','shape-rejection'),('wrong-gradient','gradient-central-difference'),
 ('sequential-update','simultaneous-update'),('sum-loss','loss-convention'),
 ('ridge-alpha','ridge-mean-convention'),('missing-cross-term','polynomial-cross-terms'),
 ('leaking-scaler','train-only-scaler'),('bad-r2','constant-target-policy'),
 ('negative-penalty','negative-penalty-rejection'),('predict-mutation','prediction-matrix'),('fixed-dimension','prediction-matrix')])
def test_real_mistakes_fail_the_corresponding_operation(defect,check):
    m=types.SimpleNamespace(**{k:getattr(qa,k)for k in dir(qa)if not k.startswith('_')})
    if defect=='broadcast':m.predict=lambda X,w,b:np.asarray(X)@w+b
    if defect=='wrong-gradient':m.gradients=lambda X,y,w,b,lam=0:(X.T@(X@w+b-y)/len(y),float(np.mean(X@w+b-y)))
    if defect=='sequential-update':
        def bad(X,y,w,b,eta,lam=0):
            gw,_=qa.gradients(X,y,w,b,lam);ww=w-eta*gw;_,gb=qa.gradients(X,y,ww,b,lam);return ww,b-eta*gb
        m.update=bad
    if defect=='sum-loss':m.loss=lambda X,y,w,b,lam=0:float(np.sum((X@w+b-y)**2)+lam*(w@w))
    if defect=='ridge-alpha':m.fit_ridge=lambda X,y,lam:qa.fit_ridge(X,y,lam/len(y))
    if defect=='missing-cross-term':m.polynomial_features=lambda X,d:np.concatenate([X**p for p in range(1,d+1)],axis=1)
    if defect=='leaking-scaler':
        def leak(X,Z):
            all=np.vstack([X,Z]);mu=all.mean(0);sd=all.std(0);return (X-mu)/sd,(Z-mu)/sd,mu,sd
        m.standardize_train=leak
    if defect=='bad-r2':m.metrics=lambda y,p:{**qa.metrics(y,p),'r2':float('nan')}
    if defect=='negative-penalty':m.fit_lasso=lambda X,y,lam:qa.fit_lasso(X,y,abs(lam))
    if defect=='predict-mutation':
        def mutate(X,w,b):X[:]=0;return X@w+b
        m.predict=mutate
    if defect=='fixed-dimension':m.predict=lambda X,w,b:qa.predict(X,w,b)if X.shape[1]==3 else np.zeros(len(X))
    rows=run_checks(m,7);assert next(r for r in rows if r['id']==check)['status']=='fail'
def test_unwritten_starter_fails_and_report_binds_actual_file(tmp_path):
    source=ROOT/'visual-lab/assessment/regression/starter.py';output=tmp_path/'run.json'
    proc=subprocess.run([sys.executable,str(ROOT/'tools/assess_regression.py'),'--submission',str(source),'--output',str(output)],text=True,capture_output=True)
    assert proc.returncode==1
    report=json.loads(output.read_text());assert report['passed']==0 and report['failed']==15 and report['skipped']==0
    assert report['sourceSha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert report['reportId']==sha(json.dumps(identity(report),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
def test_qa_fixture_is_not_labeled_a_learner_submission(tmp_path):
    source=ROOT/'visual-lab/tests/reference/learner_regression_qa.py';output=tmp_path/'qa.json'
    proc=subprocess.run([sys.executable,str(ROOT/'tools/assess_regression.py'),'--submission',str(source),'--output',str(output)],text=True,capture_output=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    report=json.loads(output.read_text());assert report['sourceRole']=='qa-fixture' and report['passed']==15

@pytest.mark.parametrize('chapter',[1,22,23,80])
def test_generic_runner_executes_a_copied_src_tree_with_trusted_tests(chapter,tmp_path):
    from tools.assess_local_submission import assess,folder_for,source_hash,suite_hash
    report=assess(chapter,folder_for(chapter)/'src',qa=True)
    assert report['exitCode']==0,report['learnerOutput']
    assert report['passed']>0 and report['skipped']==report['failed']==0
    assert report['sourceRole']=='qa-fixture'
    assert report['sourceSha256']==source_hash(folder_for(chapter)/'src')
    assert report['suiteHash']==suite_hash(chapter)
    if chapter==80:assert report['referenceDependencies'][0]['path']=='49-tokenizer-from-scratch/src/byte_bpe.py'
def test_generic_runner_rejects_reference_code_as_learner_and_incomplete_sources(tmp_path):
    from tools.assess_local_submission import assess,folder_for
    with pytest.raises(ValueError,match='reference'):assess(1,folder_for(1)/'src')
    with pytest.raises(ValueError,match='complete'):assess(1,tmp_path)
