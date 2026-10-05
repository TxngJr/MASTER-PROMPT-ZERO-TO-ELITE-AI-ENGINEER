# Supplement 06 — Neuromorphic and Spiking Neural Networks

Spiking neural networks (SNNs) represent activity as events over time rather than only dense continuous activations.

## Learning objectives

- understand spike trains and firing-rate/temporal codes;
- implement a leaky integrate-and-fire (LIF) neuron;
- understand membrane leak, threshold, reset and refractory concepts;
- understand event-driven versus clock-driven simulation;
- understand surrogate-gradient training concepts;
- understand spike-timing-dependent plasticity (STDP) concepts;
- reason about latency, spike count and energy-proxy metrics;
- distinguish neuromorphic hardware goals from ordinary GPU acceleration.

## LIF model

A simple discrete-time educational LIF update:

~~~text
v_t = beta * v_(t-1) + input_t
spike_t = 1 if v_t >= threshold else 0
if spike_t: v_t <- reset
~~~

`beta` in [0,1) controls leak. Real neuron and hardware models can use different dynamics.

## Rate and temporal coding

Rate coding represents magnitude by spike frequency over a window. Temporal coding can use spike timing itself. Coding choice changes latency, robustness and information density.

## Why gradients are difficult

The hard threshold is non-differentiable and has zero derivative almost everywhere. Surrogate-gradient methods keep the hard spike in the forward pass but substitute a smooth approximate derivative during backward optimization.

This is an approximation and should be reported explicitly.

## STDP concept

A simplified biological inspiration is that synaptic change depends on relative pre/post spike timing. Exact update rules vary; distinguish local plasticity rules from backpropagation-based SNN training.

## Evaluation

Report more than task accuracy:
- time steps / decision latency;
- average spikes per neuron/sample;
- sparsity;
- memory/state size;
- energy estimate only when the measurement/model is documented;
- robustness to timing/noise.

## Common mistakes

- calling sparse spikes automatically energy-efficient without hardware measurement;
- comparing SNN accuracy at many time steps against ANN single-pass accuracy without latency accounting;
- silently changing coding window;
- confusing biological plausibility with engineering performance;
- hiding surrogate-gradient choice.
