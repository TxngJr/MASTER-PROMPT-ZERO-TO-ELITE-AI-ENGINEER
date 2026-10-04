# Chapter 56 Exercises

1. Derive LoRA parameter count for a 4096×4096 linear layer at ranks 4/8/16/64.
2. Implement A/B low-rank updates.
3. Verify B=0 makes the initial adapter a no-op.
4. Sweep rank and measure trainable ratio.
5. Merge/unmerge adapters numerically.
6. Apply LoRA to attention projections in a tiny transformer.
7. Compare q/v-only vs all-linear targeting.
8. Estimate full FT vs LoRA optimizer-state memory.
9. Quantize a toy matrix with a generic 16-level codebook and measure error.
10. Build adapter metadata that pins the base-model revision.

Challenge:
- reproduce a PEFT LoraConfig experiment with the optional requirements file
- compare full fine-tuning and LoRA on the same tiny adaptation task
