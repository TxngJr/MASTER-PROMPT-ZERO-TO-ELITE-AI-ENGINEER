"""Independent vectorized NumPy reference; input is JSON states on stdin."""
import json
import sys
import numpy as np
out = []
for s in json.load(sys.stdin):
    x = np.array([-2, -1, 0, 1, s['sampleX']], dtype=float)
    y = np.array([-3, -1, 1, 3, s['sampleY']], dtype=float)
    y += s['noise'] * np.array([.4, -.7, .1, .6, -.4])
    w = np.array([s['w'], s['w2'], s['w3'], s['w4']][:int(s['degree'])])
    X = np.column_stack([x**j for j in range(1, len(w)+1)])
    e = X @ w + s['b'] - y
    mse = np.mean(e**2)
    grad = (2/len(x)) * X.T @ e
    penalty = 0.
    if s['regularizer'] == 'ridge':
        penalty = s['lambda'] * (w @ w)
        grad += 2 * s['lambda'] * w
    if s['regularizer'] == 'lasso':
        penalty = s['lambda'] * np.abs(w).sum()
        grad += s['lambda'] * np.sign(w)
    grad_b=2*e.mean()
    next_w=w-s['eta']*grad
    next_b=s['b']-s['eta']*grad_b
    next_error=X@next_w+next_b-y
    next_penalty=s['lambda']*np.sum(next_w**2) if s['regularizer']=='ridge' else s['lambda']*np.sum(np.abs(next_w)) if s['regularizer']=='lasso' else 0.
    out.append(dict(mse=float(mse), loss=float(mse+penalty), gradient=grad.tolist(), gradB=float(grad_b),nextWeights=next_w.tolist(),nextBias=float(next_b),nextLoss=float(np.mean(next_error**2)+next_penalty)))
print(json.dumps(out))
