# Mini Project — Tiny Text-to-Text Transformer

Create synthetic source/target transformations such as:
- copy
- reverse
- reorder marked fields

Build:
- source vocabulary
- encoder
- causal decoder
- cross-attention
- shifted target inputs
- target cross-entropy
- greedy inference

Then create span-corrupted pretraining examples using sentinel tokens and compare the objective with BERT-style MLM.
