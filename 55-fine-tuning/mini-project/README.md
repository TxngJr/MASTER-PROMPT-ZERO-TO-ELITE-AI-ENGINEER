# Mini Project — Transfer Learning Audit

Use a small pretrained backbone or pretrain a tiny network first.

Compare:
1. head-only tuning
2. last-block + head tuning
3. full fine-tuning

Track:
- trainable parameters
- peak optimizer state estimate
- train loss
- validation metric
- test metric
- retained pretraining-task metric
- time per step

Explain adaptation-vs-forgetting trade-offs.
