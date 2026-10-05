# Chapter 77 — Robotics AI

## 1. Robotics Is Closed-Loop AI

A robot policy lives inside a control loop:

~~~text
sensors
↓
observation / state estimate
↓
policy / planner / controller
↓
action
↓
environment changes
↓
new sensors
~~~

This course uses simulation-first exercises; physical hardware work should follow proper supervision and manufacturer/lab safety procedures.

## 2. Learning Objectives

- distinguish observation/state/action spaces
- understand robot control-loop timing
- derive simple forward kinematics
- implement action limits
- implement proportional control
- understand state estimation
- implement behavior-cloning loss
- understand action chunking
- distinguish imitation, RL and planning
- understand vision-language-action models
- reason about sim-to-real transfer
- design robot-policy evaluation and safety gates

## 3. Observation Space

Possible observations:
- joint positions/velocities
- camera images
- depth
- tactile sensors
- force/torque
- language instruction

Raw observations are not necessarily a Markov state.

## 4. Action Space

Possible actions:
- joint position target
- joint velocity target
- torque
- end-effector delta pose
- gripper command

Policies must match the robot/controller action convention exactly.

## 5. Control Loop Frequency

For control rate f:

~~~text
period_ms = 1000 / f
~~~

Observation + inference + action dispatch should fit the intended period when operating synchronously.

## 6. Forward Kinematics

For a planar two-link arm:

~~~text
x = l1 cos(theta1) + l2 cos(theta1 + theta2)
y = l1 sin(theta1) + l2 sin(theta1 + theta2)
~~~

This maps joint configuration to end-effector position.

## 7. Inverse Kinematics

Inverse kinematics asks for joint configuration that reaches a target pose.

Solutions may be:
- non-unique
- unreachable
- near singularities

Modern learned policies may bypass explicit IK for some tasks, but geometric constraints still matter.

## 8. Jacobian Concept

Locally:

~~~text
end_effector_velocity = J(q) * joint_velocity
~~~

Jacobians connect joint-space motion to task-space motion and expose singular configurations.

## 9. Feedback Control

A proportional controller:

~~~text
u = Kp * (target - current)
~~~

Real control often adds integral/derivative/state-feedback terms, filtering and physical limits.

## 10. Action Limits

Every policy output should pass deterministic bounds before it reaches a lower-level controller.

Examples:
- joint position limits
- velocity limits
- acceleration limits
- workspace limits

A language/model prompt is not a replacement for these constraints.

## 11. State Estimation

Sensors are noisy/partial.

State estimation combines:
- current measurements
- previous belief
- dynamics

Classical tools include Kalman filters; learned representations/world models can also maintain belief state.

## 12. Behavior Cloning

Given demonstration pairs:

~~~text
(observation_t, expert_action_t)
~~~

train policy pi to imitate expert:

~~~text
L_BC = mean ||pi(o_t) - a_t||^2
~~~

for continuous actions.

## 13. Distribution Shift

Behavior cloning is trained on expert states but its own mistakes can move it into unseen states.

This compounding error motivates:
- broader demonstrations
- corrective data
- DAgger-style collection
- stronger policies/world models

## 14. Action Chunking

Instead of predicting one action at a time, a policy can predict a chunk:

~~~text
[a_t, a_{t+1}, ..., a_{t+H-1}]
~~~

This can capture temporal consistency and reduce inference frequency.

Current LeRobot policy APIs expose action queues/chunk-aware policy behavior for several policies. (see references.md)

## 15. Modern Robot Policies

LeRobot's current policy ecosystem includes ACT, SmolVLA, π0/π0.5 and other policy families, illustrating how quickly robot-learning architectures are evolving. (see references.md)

Learn the abstractions before memorizing one policy name.

## 16. Vision-Language-Action Models

VLA concept:

~~~text
images / proprioception / language instruction
↓
multimodal policy
↓
robot action tokens / continuous actions
~~~

VLA models aim to ground language/vision representations in embodied actions.

Current LeRobot documentation explicitly describes VLA-style policies and the challenges of heterogeneous robot/action/camera spaces. (see references.md)

## 17. World Models + Robotics

Chapter 76 can feed robotics:

~~~text
current observation
+ candidate action
↓
world model imagines outcome
↓
planner chooses action
~~~

Current LeRobot v0.6 release also added world-model-policy work and simulation evaluation tooling. (see references.md)

## 18. Imitation vs RL vs Planning

Behavior cloning:
- imitate demonstrations

RL:
- optimize rewards through interaction

Planning:
- search using known/learned dynamics

Practical systems often combine them.

## 19. Sim-to-Real

Simulation differs from reality in:
- appearance
- friction
- calibration
- latency
- sensor noise
- object dynamics

Strategies:
- domain randomization
- system identification
- fine-tuning on real demonstrations
- conservative validation

## 20. Safety Layers

Robotics needs deterministic safety outside the learned policy:
- action limits
- workspace limits
- collision checks where applicable
- emergency-stop mechanisms provided by hardware/lab
- timeout/watchdog
- human supervision for deployment

This chapter's implementation remains simulation-only.

## 21. Evaluation

Track:
- task success
- collision/constraint violations
- path length
- action smoothness
- completion time
- intervention rate
- inference/control-loop latency

Success rate alone can hide unsafe behavior.

## 22. Current LeRobot Deployment

Current LeRobot rollout tooling supports deploying trained policies with different rollout strategies and emphasizes checking whether selected compute can meet required control rate. (see references.md)

For this course, treat real-robot rollout as an advanced supervised lab, not a required home exercise.

## 23. From Scratch

`src/robotics.py` implements:
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

## 24. Common Mistakes

1. action convention/unit mismatch
2. learned policy output not bounded
3. control-loop latency ignored
4. simulation success assumed real-world success
5. behavior-cloning distribution shift ignored
6. task success reported without constraint violations
7. cameras/joint observations time-unsynchronized
8. action chunk executed blindly after environment changes
9. VLA semantic ability mistaken for physical reliability
10. model safety prompt used instead of deterministic controls

## 25. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 26. Checklist

- [ ] observations/actions
- [ ] control loop
- [ ] kinematics
- [ ] feedback control
- [ ] action limits
- [ ] behavior cloning
- [ ] action chunks
- [ ] VLA
- [ ] sim-to-real
- [ ] safety evaluation

## 27. What's Next

Chapter 78 compresses AI systems into edge/TinyML hardware budgets.