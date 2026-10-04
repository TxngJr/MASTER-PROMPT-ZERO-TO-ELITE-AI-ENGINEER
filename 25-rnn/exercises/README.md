# Chapter 25 Exercises

1. Trace a 3-step RNN by hand.
2. Derive all tensor shapes for B=32,T=50,D=16,H=64.
3. Show why parameters are shared across time.
4. Implement ReLU RNN in addition to tanh.
5. Add a sequence-to-one readout.
6. Plot hidden-state norms across long sequences.
7. Demonstrate exploding gradients with a simple recurrent scalar system.
8. Add gradient clipping to a PyTorch training loop.
9. Use pack_padded_sequence with unsorted variable lengths.
10. Compare unidirectional vs bidirectional classification when the full sequence is available.

Challenge:
- connect the Batch 07 autodiff Tensor to an RNN unrolled for several steps
- implement truncated BPTT on a synthetic stream
