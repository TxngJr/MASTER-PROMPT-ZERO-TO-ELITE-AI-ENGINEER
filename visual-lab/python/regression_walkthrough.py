"""Executable counterpart of the Regression browser prototype. Run from repo root."""
import json
import numpy as np

def calculate(w=1., b=0., eta=.1):
    x=np.array([-2.,-1.,0.,1.,2.])
    y=np.array([-3.,-1.,1.,3.,5.])
    pred=w*x+b
    error=pred-y
    mse=np.mean(error**2)
    grad_w=2*np.mean(error*x)
    grad_b=2*np.mean(error)
    next_w=w-eta*grad_w
    next_b=b-eta*grad_b
    next_mse=np.mean((next_w*x+next_b-y)**2)
    return dict(w=w,b=b,mse=float(mse),gradient_w=float(grad_w),gradient_b=float(grad_b),next_w=float(next_w),next_b=float(next_b),next_mse=float(next_mse))

if __name__=='__main__':
    result=calculate()
    expected=dict(mse=3.,gradient_w=-4.,gradient_b=-2.,next_w=1.4,next_b=.2,next_mse=1.36)
    for key,value in expected.items(): np.testing.assert_allclose(result[key],value,atol=1e-12,rtol=0)
    print(json.dumps(result,indent=2))
    print('PASS: prediction -> residual -> MSE -> gradient -> update -> MSE')
