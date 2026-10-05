# Batch 27 Integration — Research-to-Production Tiny LLM

## Goal

เชื่อม Chapter 79 + Chapter 80 และ Batch 01–26:

~~~text
research claim
↓
reproduction contract
↓
dataset/tokenizer/model
↓
pretraining
↓
evaluation
↓
alignment
↓
quantization
↓
serving contract
↓
audit
~~~

## Tasks

1. เลือก claim จาก architecture/training method ที่เรียน
2. สร้าง baseline และ proposed config
3. train tiny model จาก random weights
4. run >= 3 seeds เมื่อ feasible
5. ทำ one ablation
6. report result gap/uncertainty
7. run SFT + preference smoke
8. quantization reconstruction check
9. define API/deployment/monitoring
10. run final gate

## Acceptance

ต้องมี reproducibility metadata, dataset split evidence, tokenizer identity, seed/config, parameter/memory accounting, validation, alignment evidence, quantization evidence, deployment limits และ limitations

ห้ามอ้างว่า tiny local result เทียบเท่า frontier/production scale
