# Chapter 74 Solutions — Key Ideas

- visual token count depends on model preprocessing/patching, not only raw image dimensions.
- projectors map vision features into the language model interface; architectures may also use resampling/cross-attention.
- visual tokens consume context/compute, so resolution and multi-image inputs need explicit budgets.
- grounding needs spatial metrics such as IoU/precision/recall in addition to natural-language quality.
- use each model's Processor/chat template rather than assuming text-only formatting.
