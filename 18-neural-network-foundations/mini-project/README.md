# Mini Project — Neural Forward-Pass Inspector

Build a CLI that accepts architecture such as:

~~~text
10 → 32 → 16 → 4
~~~

and reports:
- every weight shape
- every bias shape
- activation shape
- parameter count per layer
- total parameters
- min/max/mean/std of activations
- output logits
- optional sigmoid/softmax probabilities

Required experiments:
1. zero input
2. random normal input
3. huge magnitude input
4. different initialization scales
5. network with no nonlinear activations

Explain why linear-only deep networks collapse to one linear map.
