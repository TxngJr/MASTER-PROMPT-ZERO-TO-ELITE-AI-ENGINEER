# Batch 12 Integration Project — BERT vs GPT vs T5 Lab

This lab trains three tiny Transformer families under deliberately different objectives.

## Part A — Tiny BERT

~~~text
masked token sequence
↓
bidirectional encoder
↓
vocabulary logits at every position
↓
MLM loss on selected positions only
~~~

Checks:
- full encoder attention
- masked positions
- ignore_index behavior
- finite MLM loss

## Part B — Tiny GPT

~~~text
token prefix
↓
causal decoder-only block
↓
next-token logits
↓
causal LM cross-entropy
↓
autoregressive generation
~~~

Checks:
- causal attention
- shifted labels
- next-token loss
- bounded generation loop

## Part C — Tiny T5-Style Seq2Seq

Synthetic task:

~~~text
source sequence
↓ encoder
memory
↓
causal decoder + cross-attention
↓
reversed source sequence
~~~

Checks:
- encoder/decoder separation
- target shift-right
- target causal mask
- cross-attention path
- finite seq2seq loss

## Run

~~~bash
python integration-project-batch12/src/bert_gpt_t5_lab.py --mode bert --steps 20
python integration-project-batch12/src/bert_gpt_t5_lab.py --mode gpt --steps 20
python integration-project-batch12/src/bert_gpt_t5_lab.py --mode t5 --steps 20
~~~

## Methodology

These are architecture/objective labs, not benchmark-quality pretrained models.

They use:
- tiny dimensions
- internal/synthetic data
- short runs
- deterministic seeds

The purpose is to verify that the correct information flow and loss are used for each model family.

## Required Extensions

1. BERT classifier fine-tuning
2. MLM 80/10/10 corruption measurement
3. GPT top-k/top-p generation
4. GPT tied embeddings
5. GPT cached decoding
6. T5 span-corrupted pretraining examples
7. T5 greedy autoregressive decoder
8. beam-search implementation
9. source/target padding masks
10. compare parameter counts at equal d_model/depth
