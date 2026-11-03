"""QA oracle adapter, NOT a learner submission. Never grants learner credit."""
import numpy as np
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures

def data(X,y=None,w=None):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or not X.shape[0]or not X.shape[1]or not np.isfinite(X).all():raise ValueError('X must be finite nonempty (n,d)')
    if y is not None:
        y=np.asarray(y,dtype=float)
        if y.shape!=(X.shape[0],)or not np.isfinite(y).all():raise ValueError('y shape')
    if w is not None:
        w=np.asarray(w,dtype=float)
        if w.shape!=(X.shape[1],)or not np.isfinite(w).all():raise ValueError('w shape')
    return X,y,w

def penalty(value):
    if not np.isfinite(value)or value<0:raise ValueError('negative/nonfinite penalty')
def bias(value):
    if np.ndim(value)!=0 or not np.isfinite(value):raise ValueError('bias must be finite scalar')

def predict(X,w,b):
    X,_,w=data(X,w=w);bias(b)
    return X@w+b

def loss(X,y,w,b,lam=0.):
    penalty(lam);X,y,w=data(X,y,w);bias(b)
    return float(np.mean((X@w+b-y)**2)+lam*(w@w))

def gradients(X,y,w,b,lam=0.):
    penalty(lam);X,y,w=data(X,y,w);bias(b);e=X@w+b-y
    return 2*X.T@e/len(y)+2*lam*w,float(2*np.mean(e))

def update(X,y,w,b,eta,lam=0.):
    penalty(eta);gw,gb=gradients(X,y,w,b,lam)
    return np.asarray(w)-eta*gw,b-eta*gb

def fit_gd(X,y,eta=.05,steps=2000):
    X,y,_=data(X,y);w=np.zeros(X.shape[1]);b=0.
    for _ in range(steps):w,b=update(X,y,w,b,eta)
    return w,b

def fit_ols(X,y):
    X,y,_=data(X,y);z=np.linalg.lstsq(np.column_stack([X,np.ones(len(X))]),y,rcond=None)[0]
    return z[:-1],float(z[-1])

def fit_ridge(X,y,lam):
    penalty(lam);X,y,_=data(X,y)
    if lam==0:return fit_ols(X,y)
    mx=X.mean(0);my=y.mean();C=X-mx
    w=np.linalg.solve(C.T@C+len(y)*lam*np.eye(X.shape[1]),C.T@(y-my))
    return w,float(my-mx@w)

def polynomial_features(X,degree):
    X,_,_=data(X)
    if not isinstance(degree,int)or degree<1:raise ValueError('degree')
    return PolynomialFeatures(degree,include_bias=False).fit_transform(X)

def metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or not len(y):raise ValueError('shape')
    e=pred-y;sse=float(e@e);sst=float(np.sum((y-y.mean())**2));mse=float(np.mean(e*e))
    return dict(mse=mse,mae=float(np.mean(abs(e))),rmse=float(np.sqrt(mse)),r2=1-sse/sst if sst else float(sse==0))

def standardize_train(X_train,X_other):
    X,_,_=data(X_train);Z,_,_=data(X_other);mean=X.mean(0);scale=X.std(0);scale[scale==0]=1
    return (X-mean)/scale,(Z-mean)/scale,mean,scale

def fit_lasso(X,y,lam):
    penalty(lam)
    if lam==0:return fit_ols(X,y)
    model=Lasso(alpha=lam/2,tol=1e-12,max_iter=50000).fit(X,y)
    return model.coef_,float(model.intercept_)

def fit_elastic_net(X,y,l1,l2):
    penalty(l1);penalty(l2);alpha=l1/2+l2
    if alpha==0:return fit_ols(X,y)
    model=ElasticNet(alpha=alpha,l1_ratio=(l1/2)/alpha,tol=1e-12,max_iter=50000).fit(X,y)
    return model.coef_,float(model.intercept_)
