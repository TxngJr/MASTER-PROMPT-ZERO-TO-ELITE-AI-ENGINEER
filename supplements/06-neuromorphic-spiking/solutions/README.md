# Neuromorphic and Spiking Solutions

1. A spike train is a sequence of discrete spike events across time.
2. Membrane potential is the neuron state that integrates/leaks input before thresholding.
3. LIF leak decays previous state, here controlled by beta.
4. A surrogate gradient replaces the non-useful derivative of a hard spike threshold during backward optimization with a smooth approximation.
5. Rate coding uses count/frequency; temporal coding uses timing information and may trade longer observation windows against precision.
6. After crossing threshold the model emits a spike and applies its defined reset rule.
7. Without spikes and constant current I: v_t=beta^t v_0 + I * sum_(k=0)^(t-1) beta^k; for beta<1 this approaches I/(1-beta).
8. The threshold is discontinuous; its derivative is zero almost everywhere and undefined at threshold, so ordinary gradient flow is not useful.
9–11. See `src/spiking.py`; validate beta, threshold/reset and binary spike invariants.
12. Plot membrane state and event markers; explain accumulation, threshold and reset.
13. A reset-before-threshold or incorrect state reuse changes spike timing; use a tiny hand-calculated current sequence as a regression fixture.
14. Larger beta retains more previous voltage, generally integrating input over a longer effective time.
15. Run repeated fixed-seed noise trials and report spike-count/rate variability plus task metric if present.
16. Dense simulation cost scales with all neurons/time steps; an event proxy counts actual spike-driven operations but is not a hardware energy measurement.
17. Match dataset/task and report accuracy plus time steps/latency/spikes; do not compare accuracy alone.
18. Record the forward threshold and exact surrogate derivative/width used in backward; sweep it as an ablation.
19. Measure power/energy on the actual target hardware with a documented baseline, workload, sampling method and runtime; spike count alone is insufficient.
20. Include coding scheme, neuron equations/parameters, simulation step, training method, time window, accuracy/error slices, spike stats, latency/resource measurement and limitations.
