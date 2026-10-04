# Chapter 36 Solutions — Key Ideas

- encoder self-attention is bidirectional over valid source tokens.
- decoder self-attention is causal over the target prefix.
- cross-attention uses decoder queries and encoder keys/values.
- teacher forcing shifts target tokens to create decoder inputs.
- T5 span corruption replaces source spans with sentinels and asks the decoder to generate the removed spans in sentinel order.
