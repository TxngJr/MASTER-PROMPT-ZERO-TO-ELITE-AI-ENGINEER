# Chapter 50 Solutions — Key Ideas

- causal LM targets are input tokens shifted one position left.
- perplexity is exp(mean CE), so tokenizer/evaluation protocol must stay fixed for fair comparison.
- gradient accumulation should preserve the intended large-batch average gradient.
- warmup controls early optimization while cosine decay reduces LR later.
- a resumable checkpoint needs optimizer/training state in addition to model weights.
