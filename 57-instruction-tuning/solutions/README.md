# Chapter 57 Solutions — Key Ideas

- chat serialization is part of the model interface and must match the base model's expected template.
- assistant-only labels use ignore values for prompt/system/user tokens while supervising desired assistant content.
- training-time templates should contain the real response rather than appending a generation-only prompt marker.
- truncation/packing must preserve label alignment and at least one supervised target token.
- held-out instruction correctness matters more than training loss alone.
