# Batch 26 Review — Chapters 76–78

## Chapters

- 76 — World Models
- 77 — Robotics AI
- 78 — Edge AI / TinyML

## World Model Skills

- latent state / belief state
- action-conditioned dynamics
- one-step prediction
- multi-step imagination
- discounted return
- model-predictive planning
- receding-horizon control concepts
- pixel vs latent prediction
- JEPA-style representation prediction
- Dreamer-style latent imagination
- uncertainty / model exploitation
- horizon-dependent evaluation

## Robotics Skills

- observation/action spaces
- control-loop timing
- forward kinematics
- Jacobian / IK concepts
- deterministic action limits
- proportional feedback control
- state estimation concepts
- behavior cloning
- distribution shift
- action chunking
- VLA concepts
- sim-to-real
- safety metrics / intervention metrics

## Edge / TinyML Skills

- parameter/tensor storage
- Flash vs runtime RAM
- activation/tensor-arena budgeting
- memory safety margin
- operator support
- execution providers
- INT8 affine quantization
- pruning/distillation/quantization combinations
- latency / cold-start profiling
- energy per inference
- sensor windows / duty cycle
- static/ring-buffer memory planning
- mobile/IoT deployment concepts

## Implemented From Scratch

### Chapter 76
- one_step_mse
- fit_linear_dynamics
- linear_step
- rollout_linear_dynamics
- discounted_return
- goal_reward
- enumerate_action_plans
- plan_by_model
- jepa_prediction_loss

### Chapter 77
- planar_two_link_fk
- clamp_action
- proportional_control
- behavior_cloning_mse
- action_chunk
- trajectory_path_length
- action_smoothness
- goal_success
- control_period_ms
- latency_budget_ok

### Chapter 78
- tensor_storage_bytes
- model_parameter_bytes
- peak_memory_bytes
- fits_memory_budget
- operator_coverage
- affine_int8_quantize
- affine_int8_dequantize
- energy_per_inference_mj
- inferences_per_joule
- sensor_window_samples
- duty_cycle

## Integration Project

The Batch 26 integration connects:

1. world-model candidate planning in a tiny numerical environment
2. simulation-only robot action clipping and timing gates
3. edge memory / operator / INT8 / energy deployment checks
4. one final system gate requiring prediction, control safety and deployment feasibility

## Current Ecosystem Audit

### World Models

Meta's V-JEPA 2 is currently presented as world-model research focused on predictive representations and physical-world/robotics applications.

### Robotics

Current Hugging Face LeRobot policy documentation includes ACT, SmolVLA, pi0-family and other policy families. Its rollout tooling supports multiple deployment strategies and explicitly asks users to verify that the selected device can meet the required control rate.

### Edge AI

Current ONNX Runtime Mobile guidance covers CPU plus platform-specific execution providers such as XNNPACK, NNAPI and CoreML, and recommends measuring app binary size, model size, latency and power.

Current ONNX Runtime IoT/edge guidance also emphasizes local privacy/offline/latency benefits while noting model-size and compute limitations.

Current ONNX Runtime documentation states reduced-operator prebuilt mobile packages are no longer provided from 1.19 onward; use full operator support or a custom build.

## Methodology Audit

### World Models
- one-step error is not treated as sufficient long-horizon validation
- imagined reward is distinguished from real/simulator reward
- planning is receding-horizon rather than blindly open-loop
- model exploitation and uncertainty are explicit failure modes

### Robotics
- all required labs are simulation-only
- learned outputs pass deterministic action limits
- control-loop timing is part of the policy contract
- task success is not reported without constraint/safety metrics
- real-robot rollout is not required for course completion

### Edge
- model file bytes are separated from peak runtime RAM
- operator coverage is checked before accelerator assumptions
- quantized artifacts must be reevaluated
- latency and power are combined into energy/inference

## Framework Smoke

Batch 26 ONNX Runtime smoke:
- creates a tiny ONNX MatMul graph
- validates it with ONNX checker
- runs it with CPUExecutionProvider
- compares output against NumPy

No external checkpoint or device-specific accelerator is required.

## Exit Gate

Before Chapter 79:

1. Batch 26 Core CI passes
2. Batch 26 ONNX Runtime smoke passes
3. fit and roll out a learned dynamics model
4. explain one-step vs multi-step error
5. perform receding-horizon planning in simulation
6. calculate forward kinematics and control-period budget
7. enforce deterministic action limits
8. explain behavior-cloning distribution shift
9. calculate model + activation + arena memory
10. audit operator coverage
11. quantize/dequantize INT8 and measure error
12. calculate energy per inference
13. pass the full imagine + safe-control + edge-deploy integration gate