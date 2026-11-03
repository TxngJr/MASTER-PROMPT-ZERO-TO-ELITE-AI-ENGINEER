#!/usr/bin/env python3
"""Run a learner's adapter in a child Python process against fresh numerical cases.
This is local execution, not a security sandbox or a guarantee of understanding.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, io, json, os, platform
from pathlib import Path
import subprocess, sys, traceback
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'visual-lab/course/authored/mastery-policy.json'
CHECK_IDS=list(json.loads(POLICY.read_text())['regressionChecks'])
def sha(data):return hashlib.sha256(data).hexdigest()
def suite_hash():return sha(Path(__file__).read_bytes()+POLICY.read_bytes())
def identity(report):
    keys=['sourceSha256','suiteHash','seed','testedAt','sourceRole','contentVersion','chapter','suiteFamily','python','exitCode','passed','failed','skipped']
    return {**{k:report[k]for k in keys},'checks':[{'id':x['id'],'status':x['status']}for x in report['checks']]}
def run_checks(m, seed):
    import numpy as np
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures
    rng=np.random.default_rng(seed);X=rng.normal(size=(23,3));w=rng.normal(size=3);b=float(rng.normal()+4);y=X@rng.normal(size=3)+rng.normal(size=23)*.1+3
    results=[]
    def close(actual,expected,tol=1e-7):
        a=np.asarray(actual);e=np.asarray(expected)
        if a.shape!=e.shape:raise AssertionError(f'shape {a.shape} != {e.shape}')
        if not np.isfinite(a).all():raise AssertionError('nonfinite output')
        np.testing.assert_allclose(a,e,atol=tol,rtol=tol)
    def pair(value,d):
        if not isinstance(value,(tuple,list))or len(value)!=2:raise AssertionError('expected (w,b)')
        ww,bb=value;close(np.shape(ww),(d,));assert np.ndim(bb)==0,'bias must be scalar';return np.asarray(ww),float(bb)
    def rejects(call):
        try:call()
        except ValueError:return
        raise AssertionError('expected ValueError for invalid input; do not broadcast silently')
    def record(id,call):
        try:call();results.append({'id':id,'status':'pass'})
        except BaseException as exc:results.append({'id':id,'status':'fail','message':f'{type(exc).__name__}: {exc}'[:1800]})
    def prediction():
        xx=X.copy();ww=w.copy();close(m.predict(xx,ww,b),X@w+b);close(xx,X);close(ww,w)
        close(m.predict(X[:1].copy(),w.copy(),b),(X@w+b)[:1])
        for d in [1,2,4]:
            xx=rng.normal(size=(5+d,d));ww=rng.normal(size=d);bb=float(rng.normal());close(m.predict(xx,ww,bb),xx@ww+bb)
    def invalid_shape():
        for xx,ww in [(X,np.zeros(2)),(np.zeros((0,3)),w),(np.zeros((2,0)),np.array([])),(np.full((2,3),np.nan),w)]:rejects(lambda xx=xx,ww=ww:m.predict(xx,ww,b))
        rejects(lambda:m.predict(X,w,np.nan));rejects(lambda:m.predict(X,w,np.ones(len(X))))
        rejects(lambda:m.loss(X,y[:,None],w,b))
        rejects(lambda:m.metrics(y,y[:,None]))
    def loss():
        for lam in [0.,.23]:close(m.loss(X,y,w,b,lam),np.mean((X@w+b-y)**2)+lam*(w@w))
        close(m.loss(X,y,np.zeros(3),7.,.3),np.mean((7.-y)**2))
    def gradient():
        lam=.17;gw,gb=pair(m.gradients(X,y,w,b,lam),3);h=1e-5
        def J(ww,bb):return np.mean((X@ww+bb-y)**2)+lam*(ww@ww)
        fd=np.array([(J(w+np.eye(3)[j]*h,b)-J(w-np.eye(3)[j]*h,b))/(2*h)for j in range(3)])
        close(gw,fd,2e-6);close(gb,(J(w,b+h)-J(w,b-h))/(2*h),2e-6)
        for d in [1,2,4]:
            xx=rng.normal(size=(9,d));yy=rng.normal(size=9);ww=rng.normal(size=d);bb=float(rng.normal())
            gg,ggb=pair(m.gradients(xx,yy,ww,bb,lam),d)
            def objective(v,c):return np.mean((xx@v+c-yy)**2)+lam*(v@v)
            numerical=np.array([(objective(ww+np.eye(d)[j]*h,bb)-objective(ww-np.eye(d)[j]*h,bb))/(2*h)for j in range(d)])
            close(gg,numerical,2e-6);close(ggb,(objective(ww,bb+h)-objective(ww,bb-h))/(2*h),2e-6)
    def update():
        eta=.037;lam=.11;old_w=w.copy();xx=X.copy();ww,bb=pair(m.update(xx,y.copy(),old_w,b,eta,lam),3)
        e=X@w+b-y;close(ww,w-eta*(2*X.T@e/len(y)+2*lam*w));close(bb,b-eta*2*np.mean(e));close(old_w,w);close(xx,X)
        small=np.arange(-2.,3.).reshape(-1,1);target=2*small[:,0]+1
        ww,bb=pair(m.update(small,target,np.array([1.]),0.,.1),1);close(ww,np.array([1.4]));close(bb,.2);close(m.loss(small,target,ww,bb),1.36)
    def gd():
        xx=rng.normal(size=(70,3));target=xx@np.array([1.2,-.7,.4])+2.1
        ww,bb=pair(m.fit_gd(xx,target,eta=.05,steps=2000),3);close(xx@ww+bb,target,2e-5)
    def ols():
        xx=np.column_stack([X,X[:,0]]);ww,bb=pair(m.fit_ols(xx,y),4);oracle=LinearRegression().fit(xx,y);close(xx@ww+bb,oracle.predict(xx),1e-6)
    def ridge():
        for lam in [.0,.19]:
            ww,bb=pair(m.fit_ridge(X,y,lam),3);oracle=Ridge(alpha=len(y)*lam,solver='svd').fit(X,y)if lam else LinearRegression().fit(X,y)
            close(X@ww+bb,oracle.predict(X),1e-6);close(ww,oracle.coef_,1e-6)
    def poly():
        for degree in [1,2,3,4]:close(m.polynomial_features(X[:5],degree),PolynomialFeatures(degree,include_bias=False).fit_transform(X[:5]))
        rejects(lambda:m.polynomial_features(X,0));rejects(lambda:m.polynomial_features(X,2.5))
    def metric():
        pred=X@w+b;got=m.metrics(y,pred);e=pred-y
        for key,value in {'mse':np.mean(e*e),'mae':np.mean(abs(e)),'rmse':np.sqrt(np.mean(e*e)),'r2':1-sum(e*e)/sum((y-y.mean())**2)}.items():close(got[key],value)
    def constant():
        close(m.metrics(np.ones(4),np.ones(4))['r2'],1.);close(m.metrics(np.ones(4),np.zeros(4))['r2'],0.);rejects(lambda:m.metrics(np.array([]),np.array([])))
    def scaler():
        train=np.column_stack([X,np.ones(len(X))*7]);other=train[:4]+1000;before=train.copy();a,z,mean,scale=m.standardize_train(train,other)
        expected_scale=np.std(train,axis=0);expected_scale[expected_scale==0]=1
        close(mean,np.mean(train,axis=0));close(scale,expected_scale);close(a,(train-mean)/scale);close(z,(other-mean)/scale);close(train,before)
    def lasso():
        xx=np.vstack([np.eye(3),-np.eye(3)]);target=xx[:,0]*(2+rng.random())+7;lam=.4;ww,bb=pair(m.fit_lasso(xx,target,lam),3)
        g=2*xx.T@(xx@ww+bb-target)/len(xx);nz=np.abs(ww)>1e-6
        close(g[nz]+lam*np.sign(ww[nz]),np.zeros(nz.sum()),2e-4)
        assert np.all(np.abs(g[~nz])<=lam+2e-4),'zero coefficients violate KKT'
        assert np.sum(abs(ww)<1e-6)>=2,'expected sparsity for orthogonal irrelevant columns'
        close(np.mean(xx@ww+bb-target),0.,2e-4)
    def elastic():
        for l1,l2 in [(.07,.13),(0.,.1),(.1,0.)]:
            ww,bb=pair(m.fit_elastic_net(X,y,l1,l2),3);alpha=l1/2+l2
            oracle=ElasticNet(alpha=alpha,l1_ratio=(l1/2)/alpha,tol=1e-11,max_iter=30000).fit(X,y)
            close(ww,oracle.coef_,2e-4);close(bb,oracle.intercept_,2e-4)
    def penalties():
        rejects(lambda:m.loss(X,y,w,b,-.1));rejects(lambda:m.gradients(X,y,w,b,-.1));rejects(lambda:m.fit_ridge(X,y,-.1));rejects(lambda:m.fit_lasso(X,y,-.1));rejects(lambda:m.fit_elastic_net(X,y,-.1,.1));rejects(lambda:m.update(X,y,w,b,-.1))
    functions=[prediction,invalid_shape,loss,gradient,update,gd,ols,ridge,poly,metric,constant,scaler,lasso,elastic,penalties]
    for id,fn in zip(CHECK_IDS,functions,strict=True):record(id,fn)
    return results

def worker(source,seed):
    # Keep arbitrary learner prints out of the machine-readable result.
    captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
            spec=importlib.util.spec_from_file_location('learner_submission',source)
            if not spec or not spec.loader:raise ValueError('Cannot import submission')
            mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
            rows=run_checks(mod,seed)
    except BaseException as exc:rows=[{'id':id,'status':'fail','message':f'import/runtime failure: {type(exc).__name__}: {exc}'[:1800]}for id in CHECK_IDS]
    return {'checks':rows,'learnerOutput':captured.getvalue()[-5000:]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submission',type=Path,required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if not 0<=args.seed<2**32:p.error('seed must be integer in 0..2**32-1')
    source=args.submission.resolve()
    if not source.is_file()or source.suffix!='.py':p.error('submission must be an existing .py file')
    if args.worker:print(json.dumps(worker(source,args.seed),ensure_ascii=False));return 0
    if not args.output:p.error('--output is required')
    if args.output.resolve()==source:p.error('output must not overwrite submission')
    initial_hash=sha(source.read_bytes());tested_at=datetime.now(timezone.utc).isoformat()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    try:
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--submission',str(source),'--seed',str(args.seed)],text=True,capture_output=True,timeout=60,env=env)
        result=json.loads(proc.stdout)if proc.returncode==0 else {'checks':[],'learnerOutput':proc.stderr[-5000:]}
    except (subprocess.TimeoutExpired,json.JSONDecodeError)as exc:result={'checks':[],'learnerOutput':str(exc)}
    if [x.get('id')for x in result.get('checks',[])]!=CHECK_IDS:result['checks']=[{'id':id,'status':'fail','message':'worker did not finish the complete suite'}for id in CHECK_IDS]
    changed=initial_hash!=sha(source.read_bytes())
    if changed:
        for row in result['checks']:row.update(status='fail',message='source changed during assessment; rerun on a stable file')
    passed=sum(x['status']=='pass'for x in result['checks']);failed=len(CHECK_IDS)-passed
    content=json.loads((ROOT/'visual-lab/course/contracts.json').read_text())['6']['contentVersion']
    role='qa-fixture'if source.is_relative_to(ROOT/'visual-lab/tests')else'learner-submission'
    report={'schemaVersion':1,'kind':'learner-code-run','suiteFamily':'regression-challenge','chapter':6,'contentVersion':content,'sourceRole':role,'sourcePath':str(source),'sourceSha256':initial_hash,'suiteHash':suite_hash(),'seed':args.seed,'testedAt':tested_at,'python':platform.python_version(),'command':[sys.executable,str(Path(__file__).resolve()),'--submission',str(source),'--seed',str(args.seed)],'exitCode':0 if not failed else 1,'passed':passed,'failed':failed,'skipped':0,'checks':result['checks'],'learnerOutput':result.get('learnerOutput',''),'verification':'Executed locally on this file hash. Finite tests do not establish understanding, authorship, reviewer identity or all possible inputs. This JSON is not a signed attestation.'}
    report['reportId']=sha(json.dumps(identity(report),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'{passed}/{len(CHECK_IDS)} checks passed; failed={failed}, skipped=0; sourceRole={role}')
    for row in result['checks']:
        if row['status']!='pass':print(row['id']+': '+row.get('message','failed'))
    return report['exitCode']
if __name__=='__main__':sys.exit(main())
