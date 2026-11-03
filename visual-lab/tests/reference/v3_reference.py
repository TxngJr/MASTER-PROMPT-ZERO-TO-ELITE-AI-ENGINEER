"""Independent CPU oracles; framework dependency is required, never importorskip."""
import sys,json,math
import numpy as np
import torch
from sklearn.cluster import DBSCAN
for case in json.load(sys.stdin):
 s,r=case['state'],case['result']
 if case['id']=='dbscan':
  X=np.array([[0,0],[0,.2],[.2,0],[.5,0],[3,3],[3,3.2],[3.2,3],[8,8]])
  m=DBSCAN(eps=s['epsilon'],min_samples=s['minSamples']).fit(X)
  np.testing.assert_equal(m.labels_,r['labels'])
  np.testing.assert_equal(np.isin(np.arange(8),m.core_sample_indices_),r['core'])
 elif case['id']=='multihead':
  X=torch.tensor([[s['x00'],0],[0,1],[1,1]],dtype=torch.float64)
  Q=X@torch.diag(torch.tensor([s['wq'],1.],dtype=torch.float64))
  K=X@torch.diag(torch.tensor([s['wk'],1.],dtype=torch.float64))
  V=X@torch.diag(torch.tensor([s['wv'],1.],dtype=torch.float64))
  h=s['heads'];d=2//h
  q,k,v=[a.reshape(3,h,d).transpose(0,1) for a in [Q,K,V]]
  out=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=s['mask']=='causal').transpose(0,1).reshape(3,2)
  np.testing.assert_allclose(out.numpy(),r['out'],atol=1e-12)
 else:
  values=np.array([float(v) for v in s['scores'].split(',')]);state=s['seed'];indices=[]
  for _ in range(s['repeats']):
   ids=[]
   for i in range(len(values)):
    state=(1664525*state+1013904223)%2**32;ids.append(math.floor(state/2**32*len(values)))
   indices.append(ids)
  estimates=values[indices].mean(axis=1);tail=(1-s['confidenceLevel'])/2
  np.testing.assert_array_equal(indices,r['indices']);np.testing.assert_allclose(estimates,r['estimates'])
  np.testing.assert_allclose(np.quantile(estimates,[tail,1-tail]),[r['lo'],r['hi']])
print('Independent NumPy/sklearn/Torch CPU oracles passed')
