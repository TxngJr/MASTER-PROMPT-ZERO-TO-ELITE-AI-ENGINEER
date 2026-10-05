# Mini Project — Paper Reproduction Project

เลือก paper ที่ scope เหมาะกับเครื่อง local แล้ว reproduce claim หลักอย่างน้อยหนึ่ง claim

Suggested: attention ablation, optimizer comparison, small Transformer component, LoRA experiment, retrieval/reranking

## Deliverables

~~~text
paper.md
claim_table.md
reproduction_contract.yaml
src/
tests/
configs/
results/
report.md
~~~

## Required Experiments

1. baseline
2. proposed method
3. >= 3 seeds เมื่อ feasible
4. one controlled ablation
5. result-gap analysis
6. compute/hardware report

## Report

Paper/claim → differences → implementation/tests → protocol → results/uncertainty → ablation → failure analysis → limitations → conclusion

Local result ห้ามอ้างเทียบเท่า original-scale benchmark หาก data/model/compute ต่าง
