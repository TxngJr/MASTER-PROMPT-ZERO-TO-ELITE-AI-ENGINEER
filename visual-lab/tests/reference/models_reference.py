"""Vectorized independent calculations; deliberately does not import course implementations."""
import json,sys
import numpy as np

def softmax(x):
    x=np.asarray(x,dtype=float);e=np.exp(x-np.max(x));return e/e.sum()
def reference(req):
    s=req['state'];kind=req['id']
    if kind=='neuron':
        z=s['w']*s['x']+s['b'];h=max(z,0) if s['activation']=='relu' else np.tanh(z) if s['activation']=='tanh' else 1/(1+np.exp(-z))
        local=float(z>0) if s['activation']=='relu' else 1-h*h if s['activation']=='tanh' else h*(1-h)
        pred=s['v']*h;e=pred-s['target'];gz=2*e*s['v']*local
        return dict(pred=pred,loss=e*e,gw=gz*s['x'],gb=gz,gv=2*e*h)
    if kind=='attention':
        Q=np.array([s['q1'],s['q2']]);K=np.array([[s['k11'],s['k12']],[0,1],[1,1],[-1,1]])
        scores=K@Q/np.sqrt(2)/s['temperature'];positions=np.arange(4)
        mask=np.ones(4,bool) if s['mask']=='none' else positions<=s['queryPosition']
        if s['mask']=='window':mask &= positions>=s['queryPosition']-s['window']+1
        scores[~mask]=-np.inf;p=softmax(scores)
        return dict(weights=p.tolist(),out=p@np.array([s['v1'],2,3,4]))
    if kind=='convolution':
        X=np.tile([0.,0,1,1,1],(5,1));X[2,2]=s['pixel'];X=np.pad(X,s['padding'])
        K=np.array([[-1.,0,s['kernelWeight']],[-1,0,1],[-1,0,1]])
        patches=np.lib.stride_tricks.sliding_window_view(X,(3,3))[::s['stride'],::s['stride']]
        Y=np.einsum('ijab,ab->ij',patches,K)+s['bias']
        return dict(out=Y.tolist(),selected=Y.flat[s['position']%Y.size])
    if kind=='lora':
        r=s['rank'];A0=np.fromfunction(lambda k,j:(k+j+1)/4,(r,4));B0=np.fromfunction(lambda i,k:(i+k)%3-1,(4,r));x=np.array([1.,-1,.5,0])
        C=(s['alpha']/r)*(B0@A0);out=(np.eye(4)+s['a']*s['b']*C)@x;e=out[0]-s['target'];base=(C@x)[0]
        return dict(loss=e*e,ga=2*e*s['b']*base,gb=2*e*s['a']*base)
    if kind=='preference':
        p=softmax([s['p0'],s['p1']]);ref=softmax([s['reference'],0]);kl=np.sum(p*np.log(p/ref))
        z=s['beta']*((s['p0']-s['p1'])-s['reference']);loss=np.logaddexp(0,-z)
        g=s['beta']*(-1/(1+np.exp(z))) if s['mode']=='dpo' else -(s['reward0']*p[0]*p[1]-s['beta']*p[0]*p[1]*np.log(p[0]*ref[1]/(p[1]*ref[0])))
        if s['mode']=='rlhf':loss=-p[0]*s['reward0']+s['beta']*kl
        return dict(loss=loss,gradient=g,kl=kl,p=p.tolist())
    if kind=='queue':
        ready=np.zeros(s['workers']);rows=[]
        for i in range(s['jobs']):
            arrival=i*s['arrival'];w=int(np.argmin(ready));start=max(arrival,ready[w]);end=start+s['service'];ready[w]=end
            rows.append([i,arrival,w,start,end,start-arrival,end-arrival])
        return dict(rows=rows)
    raise ValueError(kind)
print(json.dumps([reference(r) for r in json.load(sys.stdin)]))
