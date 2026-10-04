# Mini Project — Synthetic Community Node Classifier

Generate a graph with several communities.

Create:
- noisy node features
- within-community and cross-community edges
- train/validation/test node masks

Compare:
- MLP ignoring graph
- one-layer GCN
- two-layer GCN
- GraphSAGE-style mean model

Report:
- accuracy
- parameter count
- depth
- homophily level
- oversmoothing diagnostics
