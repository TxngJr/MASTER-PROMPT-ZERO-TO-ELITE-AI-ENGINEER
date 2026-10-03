# Chapter 13 Solutions — Key Ideas

- boundary: `wᵀx+b=0`
- signed distance: `(wᵀx+b)/||w||`
- hard margin constraints: `y_i f(x_i)>=1`
- hinge: `max(0,1-yf(x))`
- C larger generally means violations are penalized more strongly
- RBF: `exp(-gamma||x-z||²)`
- gamma high generally produces more local, flexible influence

Do not tune C/gamma on test data.
