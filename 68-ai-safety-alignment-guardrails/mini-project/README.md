# Mini Project — Guarded Tool-Calling RAG System

Using only mock/non-destructive tools, build a pipeline with:
- authenticated principal/groups
- ACL-filtered retrieval
- input/request budget
- tool allow-list
- read/write distinction
- explicit approval for writes
- strict tool-argument schema
- output validation
- decision/audit metadata
- held-out safety + benign eval set

Report block recall, false-positive rate and which deterministic layer stopped each unsafe test case.
