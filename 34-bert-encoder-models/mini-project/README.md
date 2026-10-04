# Mini Project — Tiny Masked Encoder

Create a tiny internal corpus.

Build:
- vocabulary
- CLS/SEP/MASK/PAD tokens
- MLM corruption pipeline
- bidirectional Transformer encoder
- vocabulary prediction head

Track:
- MLM loss
- masked-token accuracy
- percentage selected
- percentage actually replaced with MASK/random/unchanged

Then add a small classification head and fine-tune on a synthetic sequence-label task.
