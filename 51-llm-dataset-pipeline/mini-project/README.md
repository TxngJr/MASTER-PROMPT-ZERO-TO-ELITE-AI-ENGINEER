# Mini Project — Reproducible Pretraining Corpus Builder

Build a local corpus pipeline:

1. ingest Markdown/text records
2. normalize
3. exact deduplicate
4. assign document-level splits
5. train/freeze tokenizer on training corpus only
6. tokenize each split
7. insert EOS/document separators
8. pack fixed-length sequences
9. shard arrays
10. generate manifest/checksums

Validation must prove train/validation document IDs and hashes do not overlap.
