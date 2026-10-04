# Batch 26 Integration Project — Imagine, Act Safely & Deploy at the Edge

Batch 26 connects learned predictive models, simulation-first robotics control and edge deployment constraints.

~~~text
learn / use world model
↓
imagine candidate plans
↓
choose simulated action plan
↓
deterministic robot action limits
↓
control-loop latency gate
↓
edge memory/operator/quantization gate
~~~

## Part A — World Model

A simple 1D learned-model-style planner evaluates candidate action sequences and selects a plan reaching a target.

Report:
- chosen action sequence
- imagined trajectory
- predicted return
- final state

## Part B — Robotics

The robotics section is simulation-only.

It checks:
- proportional action proposal
- deterministic action clipping
- two-link forward kinematics
- 50 Hz control-period budget

## Part C — Edge Deployment

Estimate:
- FP32 vs INT8 parameter bytes
- peak runtime memory
- 64 KiB memory-budget fit
- operator coverage
- INT8 round-trip error
- energy per inference

## Full Gate

The full system passes only when:
- world-model plan reaches the simulated goal
- actions remain inside deterministic bounds and timing budget
- edge artifact fits memory and supported-operator constraints

## Run

~~~bash
python integration-project-batch26/src/world_robot_edge_lab.py --mode world
python integration-project-batch26/src/world_robot_edge_lab.py --mode robotics
python integration-project-batch26/src/world_robot_edge_lab.py --mode edge
python integration-project-batch26/src/world_robot_edge_lab.py --mode full
~~~

## ONNX Runtime Smoke

A dedicated CI test creates a tiny ONNX MatMul graph in memory, validates it, runs it with CPUExecutionProvider and compares the result with NumPy.

No external model checkpoint or physical device is needed.

## Required Extensions

1. train a tiny latent dynamics model on synthetic trajectories
2. compare one-step and multi-step rollout error
3. implement receding-horizon planning in simulation
4. train a small behavior-cloning policy in simulation
5. add action bounds/timeouts to the simulated controller
6. export a tiny trained model to ONNX
7. audit operators before deployment
8. compare FP32 and INT8 error/bytes
9. benchmark warm latency/energy on a safe target device
10. produce a final quality-memory-latency-energy report