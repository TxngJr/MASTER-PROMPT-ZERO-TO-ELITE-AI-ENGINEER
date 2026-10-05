# Chapter 28 — GAN Solutions
1. The generator maps a latent/noise input to a synthetic sample in the data domain.
2. The discriminator maps a sample to a score/probability used to distinguish data-source classes during training.
3. The minimax objective couples both networks so each improves against the current opponent; state which terms belong to real and generated samples.
4. The non-saturating form changes the generator objective to provide a stronger learning signal when generated samples are initially easy to classify.
5. Generator produces candidates; discriminator supplies the classification signal. They have different objectives but share the same training game.
6. If one side becomes much stronger, useful gradients/signals can degrade. Compare loss trends and sample quality rather than one scalar alone.
7. Mode collapse means many latent inputs map to too little output diversity. Detect it with sample diversity checks and task-appropriate statistics.
8. More capacity can improve fit but can also make imbalance and instability harder to control; compare equal-budget settings experimentally.
9. Keep the transformation explicit, validate input/output shape, and use a fixed input for repeatability.
10. Test one normal shape and one edge shape with expected output properties.
11. Compute discriminator real/fake terms and the generator term directly from logits or probabilities using a stable formulation.
12. Fix all seeds, record architecture and optimizer settings, and track both losses plus sample statistics.
13. Check batch, channel, and feature dimensions at each network boundary and correct the first mismatch.
14. Prefer stable logit-based binary cross-entropy rather than manually taking logs of probabilities near zero or one.
15. Keep held-out evaluation samples separate; training losses are not sufficient evidence of generation quality.
16. Measure one full step, identify the dominant network operation, and verify that any optimization preserves output shape and numerical behavior.
17. Trace latent input → generator → generated sample → discriminator → losses → alternating parameter updates.
18. Use the same tensors and reduction convention before comparing scratch and framework values.
19. Change one stabilization factor at a time, keep seeds/settings fixed, and compare multiple runs when randomness matters.
20. Record input constraints, model/data/config versions, quality metrics, resource use, seeds, and a falsifiable question such as how a chosen stabilization method changes diversity at fixed compute.
