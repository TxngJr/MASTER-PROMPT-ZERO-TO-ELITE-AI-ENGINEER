# Chapter Completion Addenda — Chapters 13–78

Use this companion after each chapter README. It supplies the teaching-contract sections that were not consistently labeled in the historical chapter files. Detailed theory, equations, source code and references remain in each chapter directory.

## 13-svm — Support Vector Machines

**Prerequisites:** Complete Chapters 01–12 foundations; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Support Vector Machines as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Support Vector Machines; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Support Vector Machines.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Support Vector Machines from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Support Vector Machines, and when the chapter mini-project plus twenty total exercises are complete.

## 14-clustering — Clustering

**Prerequisites:** Complete 13-svm; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Clustering as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Clustering; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Clustering.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Clustering from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Clustering, and when the chapter mini-project plus twenty total exercises are complete.

## 15-dimensionality-reduction — Dimensionality Reduction

**Prerequisites:** Complete 14-clustering; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Dimensionality Reduction as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Dimensionality Reduction; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Dimensionality Reduction.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Dimensionality Reduction from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Dimensionality Reduction, and when the chapter mini-project plus twenty total exercises are complete.

## 16-anomaly-detection — Anomaly Detection

**Prerequisites:** Complete 15-dimensionality-reduction; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Anomaly Detection as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Anomaly Detection; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Anomaly Detection.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Anomaly Detection from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Anomaly Detection, and when the chapter mini-project plus twenty total exercises are complete.

## 17-time-series — Time Series

**Prerequisites:** Complete 16-anomaly-detection; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Time Series as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Time Series; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Time Series.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Time Series from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Time Series, and when the chapter mini-project plus twenty total exercises are complete.

## 18-neural-network-foundations — Neural Network Foundations

**Prerequisites:** Complete 17-time-series; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Neural Network Foundations as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Neural Network Foundations; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Neural Network Foundations.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Neural Network Foundations from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Neural Network Foundations, and when the chapter mini-project plus twenty total exercises are complete.

## 19-backpropagation — Backpropagation

**Prerequisites:** Complete 18-neural-network-foundations; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Backpropagation as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Backpropagation; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Backpropagation.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Backpropagation from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Backpropagation, and when the chapter mini-project plus twenty total exercises are complete.

## 20-optimizers — Optimizers

**Prerequisites:** Complete 19-backpropagation; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Optimizers as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Optimizers; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Optimizers.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Optimizers from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Optimizers, and when the chapter mini-project plus twenty total exercises are complete.

## 21-activations-losses — Activations and Losses

**Prerequisites:** Complete 20-optimizers; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Activations and Losses as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Activations and Losses; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Activations and Losses.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Activations and Losses from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Activations and Losses, and when the chapter mini-project plus twenty total exercises are complete.

## 22-pytorch — PyTorch

**Prerequisites:** Complete 21-activations-losses; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat PyTorch as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to PyTorch; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for PyTorch.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain PyTorch from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate PyTorch, and when the chapter mini-project plus twenty total exercises are complete.

## 23-tensorflow-keras — TensorFlow and Keras

**Prerequisites:** Complete 22-pytorch; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat TensorFlow and Keras as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to TensorFlow and Keras; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for TensorFlow and Keras.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain TensorFlow and Keras from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate TensorFlow and Keras, and when the chapter mini-project plus twenty total exercises are complete.

## 24-cnn — Convolutional Neural Networks

**Prerequisites:** Complete 23-tensorflow-keras; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Convolutional Neural Networks as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Convolutional Neural Networks; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Convolutional Neural Networks.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Convolutional Neural Networks from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Convolutional Neural Networks, and when the chapter mini-project plus twenty total exercises are complete.

## 25-rnn — Recurrent Neural Networks

**Prerequisites:** Complete 24-cnn; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Recurrent Neural Networks as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Recurrent Neural Networks; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Recurrent Neural Networks.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Recurrent Neural Networks from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Recurrent Neural Networks, and when the chapter mini-project plus twenty total exercises are complete.

## 26-lstm-gru — LSTM and GRU

**Prerequisites:** Complete 25-rnn; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LSTM and GRU as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LSTM and GRU; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LSTM and GRU.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LSTM and GRU from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LSTM and GRU, and when the chapter mini-project plus twenty total exercises are complete.

## 27-autoencoder-vae — Autoencoders and VAE

**Prerequisites:** Complete 26-lstm-gru; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Autoencoders and VAE as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Autoencoders and VAE; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Autoencoders and VAE.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Autoencoders and VAE from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Autoencoders and VAE, and when the chapter mini-project plus twenty total exercises are complete.

## 28-gan — GAN

**Prerequisites:** Complete 27-autoencoder-vae; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat GAN as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to GAN; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for GAN.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain GAN from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate GAN, and when the chapter mini-project plus twenty total exercises are complete.

## 29-attention — Attention

**Prerequisites:** Complete 28-gan; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Attention as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Attention; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Attention.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Attention from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Attention, and when the chapter mini-project plus twenty total exercises are complete.

## 30-transformer — Transformer

**Prerequisites:** Complete 29-attention; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Transformer as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Transformer; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Transformer.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Transformer from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Transformer, and when the chapter mini-project plus twenty total exercises are complete.

## 31-vision-transformer — Vision Transformer

**Prerequisites:** Complete 30-transformer; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Vision Transformer as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Vision Transformer; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Vision Transformer.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Vision Transformer from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Vision Transformer, and when the chapter mini-project plus twenty total exercises are complete.

## 32-nlp-fundamentals — NLP Fundamentals

**Prerequisites:** Complete 31-vision-transformer; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat NLP Fundamentals as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to NLP Fundamentals; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for NLP Fundamentals.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain NLP Fundamentals from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate NLP Fundamentals, and when the chapter mini-project plus twenty total exercises are complete.

## 33-word-embeddings — Word Embeddings

**Prerequisites:** Complete 32-nlp-fundamentals; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Word Embeddings as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Word Embeddings; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Word Embeddings.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Word Embeddings from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Word Embeddings, and when the chapter mini-project plus twenty total exercises are complete.

## 34-bert-encoder-models — BERT and Encoder Models

**Prerequisites:** Complete 33-word-embeddings; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat BERT and Encoder Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to BERT and Encoder Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for BERT and Encoder Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain BERT and Encoder Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate BERT and Encoder Models, and when the chapter mini-project plus twenty total exercises are complete.

## 35-gpt-decoder-models — GPT and Decoder Models

**Prerequisites:** Complete 34-bert-encoder-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat GPT and Decoder Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to GPT and Decoder Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for GPT and Decoder Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain GPT and Decoder Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate GPT and Decoder Models, and when the chapter mini-project plus twenty total exercises are complete.

## 36-encoder-decoder-t5 — Encoder-Decoder and T5

**Prerequisites:** Complete 35-gpt-decoder-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Encoder-Decoder and T5 as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Encoder-Decoder and T5; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Encoder-Decoder and T5.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Encoder-Decoder and T5 from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Encoder-Decoder and T5, and when the chapter mini-project plus twenty total exercises are complete.

## 37-graph-neural-networks — Graph Neural Networks

**Prerequisites:** Complete 36-encoder-decoder-t5; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Graph Neural Networks as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Graph Neural Networks; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Graph Neural Networks.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Graph Neural Networks from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Graph Neural Networks, and when the chapter mini-project plus twenty total exercises are complete.

## 38-reinforcement-learning — Reinforcement Learning

**Prerequisites:** Complete 37-graph-neural-networks; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Reinforcement Learning as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Reinforcement Learning; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Reinforcement Learning.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Reinforcement Learning from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Reinforcement Learning, and when the chapter mini-project plus twenty total exercises are complete.

## 39-generative-ai-foundations — Generative AI Foundations

**Prerequisites:** Complete 38-reinforcement-learning; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Generative AI Foundations as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Generative AI Foundations; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Generative AI Foundations.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Generative AI Foundations from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Generative AI Foundations, and when the chapter mini-project plus twenty total exercises are complete.

## 40-diffusion-models — Diffusion Models

**Prerequisites:** Complete 39-generative-ai-foundations; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Diffusion Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Diffusion Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Diffusion Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Diffusion Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Diffusion Models, and when the chapter mini-project plus twenty total exercises are complete.

## 41-multimodal-models — Multimodal Models

**Prerequisites:** Complete 40-diffusion-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Multimodal Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Multimodal Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Multimodal Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Multimodal Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Multimodal Models, and when the chapter mini-project plus twenty total exercises are complete.

## 42-speech-stt-tts — Speech, STT and TTS

**Prerequisites:** Complete 41-multimodal-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Speech, STT and TTS as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Speech, STT and TTS; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Speech, STT and TTS.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Speech, STT and TTS from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Speech, STT and TTS, and when the chapter mini-project plus twenty total exercises are complete.

## 43-recommender-systems — Recommender Systems

**Prerequisites:** Complete 42-speech-stt-tts; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Recommender Systems as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Recommender Systems; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Recommender Systems.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Recommender Systems from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Recommender Systems, and when the chapter mini-project plus twenty total exercises are complete.

## 44-object-detection-segmentation — Object Detection and Segmentation

**Prerequisites:** Complete 43-recommender-systems; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Object Detection and Segmentation as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Object Detection and Segmentation; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Object Detection and Segmentation.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Object Detection and Segmentation from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Object Detection and Segmentation, and when the chapter mini-project plus twenty total exercises are complete.

## 45-embeddings-vector-databases — Embeddings and Vector Databases

**Prerequisites:** Complete 44-object-detection-segmentation; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Embeddings and Vector Databases as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Embeddings and Vector Databases; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Embeddings and Vector Databases.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Embeddings and Vector Databases from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Embeddings and Vector Databases, and when the chapter mini-project plus twenty total exercises are complete.

## 46-rag — Retrieval-Augmented Generation

**Prerequisites:** Complete 45-embeddings-vector-databases; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Retrieval-Augmented Generation as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Retrieval-Augmented Generation; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Retrieval-Augmented Generation.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Retrieval-Augmented Generation from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Retrieval-Augmented Generation, and when the chapter mini-project plus twenty total exercises are complete.

## 47-ai-agents-tool-calling — AI Agents and Tool Calling

**Prerequisites:** Complete 46-rag; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat AI Agents and Tool Calling as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to AI Agents and Tool Calling; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for AI Agents and Tool Calling.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain AI Agents and Tool Calling from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate AI Agents and Tool Calling, and when the chapter mini-project plus twenty total exercises are complete.

## 48-llm-architecture — LLM Architecture

**Prerequisites:** Complete 47-ai-agents-tool-calling; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LLM Architecture as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LLM Architecture; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LLM Architecture.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LLM Architecture from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LLM Architecture, and when the chapter mini-project plus twenty total exercises are complete.

## 49-tokenizer-from-scratch — Tokenizer From Scratch

**Prerequisites:** Complete 48-llm-architecture; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Tokenizer From Scratch as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Tokenizer From Scratch; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Tokenizer From Scratch.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Tokenizer From Scratch from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Tokenizer From Scratch, and when the chapter mini-project plus twenty total exercises are complete.

## 50-llm-pretraining-from-scratch — LLM Pretraining From Scratch

**Prerequisites:** Complete 49-tokenizer-from-scratch; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LLM Pretraining From Scratch as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LLM Pretraining From Scratch; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LLM Pretraining From Scratch.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LLM Pretraining From Scratch from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LLM Pretraining From Scratch, and when the chapter mini-project plus twenty total exercises are complete.

## 51-llm-dataset-pipeline — LLM Dataset Pipeline

**Prerequisites:** Complete 50-llm-pretraining-from-scratch; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LLM Dataset Pipeline as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LLM Dataset Pipeline; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LLM Dataset Pipeline.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LLM Dataset Pipeline from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LLM Dataset Pipeline, and when the chapter mini-project plus twenty total exercises are complete.

## 52-distributed-training — Distributed Training

**Prerequisites:** Complete 51-llm-dataset-pipeline; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Distributed Training as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Distributed Training; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Distributed Training.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Distributed Training from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Distributed Training, and when the chapter mini-project plus twenty total exercises are complete.

## 53-gpu-cuda-fundamentals — GPU and CUDA Fundamentals

**Prerequisites:** Complete 52-distributed-training; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat GPU and CUDA Fundamentals as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to GPU and CUDA Fundamentals; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for GPU and CUDA Fundamentals.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain GPU and CUDA Fundamentals from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate GPU and CUDA Fundamentals, and when the chapter mini-project plus twenty total exercises are complete.

## 54-mixed-precision — Mixed Precision

**Prerequisites:** Complete 53-gpu-cuda-fundamentals; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Mixed Precision as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Mixed Precision; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Mixed Precision.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Mixed Precision from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Mixed Precision, and when the chapter mini-project plus twenty total exercises are complete.

## 55-fine-tuning — Fine-Tuning

**Prerequisites:** Complete 54-mixed-precision; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Fine-Tuning as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Fine-Tuning; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Fine-Tuning.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Fine-Tuning from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Fine-Tuning, and when the chapter mini-project plus twenty total exercises are complete.

## 56-lora-qlora-peft — LoRA, QLoRA and PEFT

**Prerequisites:** Complete 55-fine-tuning; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LoRA, QLoRA and PEFT as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LoRA, QLoRA and PEFT; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LoRA, QLoRA and PEFT.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LoRA, QLoRA and PEFT from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LoRA, QLoRA and PEFT, and when the chapter mini-project plus twenty total exercises are complete.

## 57-instruction-tuning — Instruction Tuning

**Prerequisites:** Complete 56-lora-qlora-peft; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Instruction Tuning as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Instruction Tuning; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Instruction Tuning.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Instruction Tuning from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Instruction Tuning, and when the chapter mini-project plus twenty total exercises are complete.

## 58-rlhf — RLHF

**Prerequisites:** Complete 57-instruction-tuning; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat RLHF as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to RLHF; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for RLHF.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain RLHF from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate RLHF, and when the chapter mini-project plus twenty total exercises are complete.

## 59-dpo-preference-optimization — DPO and Preference Optimization

**Prerequisites:** Complete 58-rlhf; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat DPO and Preference Optimization as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to DPO and Preference Optimization; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for DPO and Preference Optimization.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain DPO and Preference Optimization from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate DPO and Preference Optimization, and when the chapter mini-project plus twenty total exercises are complete.

## 60-llm-evaluation — LLM Evaluation

**Prerequisites:** Complete 59-dpo-preference-optimization; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LLM Evaluation as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LLM Evaluation; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LLM Evaluation.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LLM Evaluation from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LLM Evaluation, and when the chapter mini-project plus twenty total exercises are complete.

## 61-quantization — Quantization

**Prerequisites:** Complete 60-llm-evaluation; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Quantization as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Quantization; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Quantization.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Quantization from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Quantization, and when the chapter mini-project plus twenty total exercises are complete.

## 62-inference-optimization — Inference Optimization

**Prerequisites:** Complete 61-quantization; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Inference Optimization as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Inference Optimization; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Inference Optimization.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Inference Optimization from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Inference Optimization, and when the chapter mini-project plus twenty total exercises are complete.

## 63-llm-serving — LLM Serving

**Prerequisites:** Complete 62-inference-optimization; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat LLM Serving as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to LLM Serving; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for LLM Serving.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain LLM Serving from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate LLM Serving, and when the chapter mini-project plus twenty total exercises are complete.

## 64-deploy-ai — AI Deployment

**Prerequisites:** Complete 63-llm-serving; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat AI Deployment as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to AI Deployment; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for AI Deployment.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain AI Deployment from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate AI Deployment, and when the chapter mini-project plus twenty total exercises are complete.

## 65-mlops — MLOps

**Prerequisites:** Complete 64-deploy-ai; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat MLOps as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to MLOps; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for MLOps.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain MLOps from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate MLOps, and when the chapter mini-project plus twenty total exercises are complete.

## 66-ai-data-engineering — AI Data Engineering

**Prerequisites:** Complete 65-mlops; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat AI Data Engineering as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to AI Data Engineering; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for AI Data Engineering.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain AI Data Engineering from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate AI Data Engineering, and when the chapter mini-project plus twenty total exercises are complete.

## 67-ai-distributed-systems — AI Distributed Systems

**Prerequisites:** Complete 66-ai-data-engineering; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat AI Distributed Systems as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to AI Distributed Systems; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for AI Distributed Systems.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain AI Distributed Systems from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate AI Distributed Systems, and when the chapter mini-project plus twenty total exercises are complete.

## 68-ai-safety-alignment-guardrails — AI Safety, Alignment and Guardrails

**Prerequisites:** Complete 67-ai-distributed-systems; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat AI Safety, Alignment and Guardrails as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to AI Safety, Alignment and Guardrails; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for AI Safety, Alignment and Guardrails.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain AI Safety, Alignment and Guardrails from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate AI Safety, Alignment and Guardrails, and when the chapter mini-project plus twenty total exercises are complete.

## 69-interpretability-xai — Interpretability and XAI

**Prerequisites:** Complete 68-ai-safety-alignment-guardrails; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Interpretability and XAI as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Interpretability and XAI; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Interpretability and XAI.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Interpretability and XAI from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Interpretability and XAI, and when the chapter mini-project plus twenty total exercises are complete.

## 70-model-compression — Model Compression

**Prerequisites:** Complete 69-interpretability-xai; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Model Compression as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Model Compression; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Model Compression.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Model Compression from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Model Compression, and when the chapter mini-project plus twenty total exercises are complete.

## 71-mixture-of-experts — Mixture of Experts

**Prerequisites:** Complete 70-model-compression; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Mixture of Experts as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Mixture of Experts; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Mixture of Experts.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Mixture of Experts from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Mixture of Experts, and when the chapter mini-project plus twenty total exercises are complete.

## 72-long-context-memory — Long Context and Memory

**Prerequisites:** Complete 71-mixture-of-experts; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Long Context and Memory as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Long Context and Memory; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Long Context and Memory.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Long Context and Memory from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Long Context and Memory, and when the chapter mini-project plus twenty total exercises are complete.

## 73-reasoning-models — Reasoning Models

**Prerequisites:** Complete 72-long-context-memory; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Reasoning Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Reasoning Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Reasoning Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Reasoning Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Reasoning Models, and when the chapter mini-project plus twenty total exercises are complete.

## 74-vision-language-models — Vision-Language Models

**Prerequisites:** Complete 73-reasoning-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Vision-Language Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Vision-Language Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Vision-Language Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Vision-Language Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Vision-Language Models, and when the chapter mini-project plus twenty total exercises are complete.

## 75-audio-voice-models — Audio and Voice Models

**Prerequisites:** Complete 74-vision-language-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Audio and Voice Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Audio and Voice Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Audio and Voice Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Audio and Voice Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Audio and Voice Models, and when the chapter mini-project plus twenty total exercises are complete.

## 76-world-models — World Models

**Prerequisites:** Complete 75-audio-voice-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat World Models as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to World Models; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for World Models.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain World Models from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate World Models, and when the chapter mini-project plus twenty total exercises are complete.

## 77-robotics-ai — Robotics AI

**Prerequisites:** Complete 76-world-models; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Robotics AI as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Robotics AI; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Robotics AI.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Robotics AI from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Robotics AI, and when the chapter mini-project plus twenty total exercises are complete.

## 78-edge-ai-tinyml — Edge AI and TinyML

**Prerequisites:** Complete 77-robotics-ai; be able to trace array/tensor shapes, separate train/validation/test data, and explain the core objective of the current chapter.

**Mental model:** Treat Edge AI and TinyML as a pipeline: input representation → central mechanism → objective/state update → observable output. At every boundary, state shape/state, allowed information flow, and the invariant that should remain true.

**Code walkthrough:** Read public API first, then input checks, state/shape transformations, numerical core, returned values and tests. Connect each important code block to the equation or algorithm in the chapter README.

**Visualization / inspection:** Use a tiny case. Inspect intermediate values, distributions, norms, scores, states or trajectories appropriate to Edge AI and TinyML; do not rely only on the final metric.

**Experiment:** Run a baseline, one controlled parameter change, one edge case and one repeated seeded comparison. Record seed, environment, data version, configuration and metric.

**Limitations:** List at least one assumption, one data-related limitation, one numerical/optimization limitation and one scaling limitation for Edge AI and TinyML.

**Analysis workflow:** Check inputs → preprocessing → shapes/state → finite numerical values → objective/update → evaluation protocol. The earliest unexpected state is more informative than a downstream metric.

**Performance:** Identify the dominant time and memory terms and measure them on the local learning run before attempting optimization.

**Production perspective:** Define input validation, artifact/configuration versioning, quality and resource measurements, staged release criteria and a way to return to the previous known-good artifact.

**Research perspective:** State a falsifiable hypothesis, baseline, one-factor change, metric, repeated-run plan when stochastic, and how negative results will be recorded.

**Interview questions:**
1. Explain Edge AI and TinyML from first principles without naming a library API.
2. Trace one input through the central mechanism and give the important intermediate shapes/state.
3. Name the strongest assumption and a case where it fails.
4. State the dominant compute/memory cost.
5. Describe how you would validate a new implementation against a trusted baseline.

**Summary / exit condition:** You may move on when you can explain, derive or trace, implement, test, analyze, measure and evaluate Edge AI and TinyML, and when the chapter mini-project plus twenty total exercises are complete.

