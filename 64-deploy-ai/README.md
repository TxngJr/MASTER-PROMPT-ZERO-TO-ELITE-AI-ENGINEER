# Chapter 64 — Deploy AI: API, Docker, Kubernetes & Cloud

## 1. Deployment Is a System Problem

Training creates an artifact. Deployment makes that artifact reliably reachable under real traffic.

~~~text
model artifact
↓
container image
↓
registry
↓
Kubernetes / cloud runtime
↓
service / gateway
↓
observability + autoscaling + rollout
~~~

## 2. Learning Objectives

- package an AI service into an immutable container
- understand Docker layers and image reproducibility
- separate build-time and runtime dependencies
- design health, readiness and startup probes
- understand Deployment / ReplicaSet / Pod / Service
- request GPU resources correctly
- use ConfigMap and Secret appropriately
- reason about Gateway / Ingress
- calculate rolling-update capacity
- understand HPA and custom metrics
- design canary/blue-green rollouts
- reason about cloud load balancers and registries
- define rollback and disaster-recovery criteria

## 3. API Layer

A model server should expose a stable contract rather than leaking framework internals.

Typical API responsibilities:
- request validation
- model/version selection
- authentication/authorization
- timeouts and cancellation
- response schema
- health/readiness
- tracing / metrics

## 4. Immutable Artifacts

Production deployment should identify exact:
- source commit
- model revision
- tokenizer/config
- container digest
- dependency lock

A mutable tag such as latest is insufficient for reproducible rollback.

## 5. Docker Image Layers

Container builds form cacheable layers. Put slowly changing dependency installation before frequently changing source files when practical.

Use multi-stage builds when build-time tooling does not belong in the runtime image.

## 6. Runtime Image

Prefer:
- non-root user
- minimal packages
- pinned dependencies
- explicit entrypoint
- read-only configuration where possible
- no embedded credentials

Do not bake cloud tokens, API keys or model-registry secrets into the image.

## 7. Container Resources

Define CPU and memory requests/limits based on measurements.

For GPU workloads, Kubernetes exposes vendor devices through device plugins. A typical NVIDIA resource is `nvidia.com/gpu`.

## 8. Kubernetes Object Model

~~~text
Deployment
↓ manages
ReplicaSet
↓ manages
Pods

Service
↓ stable discovery/load balancing
Pods
~~~

Deployments support declarative rollout and rollback of stateless workloads.

## 9. Service

A Service gives a stable virtual endpoint to a changing set of Pods selected by labels.

Common service types:
- ClusterIP
- NodePort
- LoadBalancer

## 10. Gateway vs Ingress

Ingress remains stable but its API is frozen. Kubernetes recommends Gateway API for new feature development.

For new systems, understand:
- GatewayClass
- Gateway
- HTTPRoute

while still being able to maintain existing Ingress deployments.

## 11. ConfigMap

Use ConfigMap for non-secret configuration such as:
- model name
- log level
- feature flags
- timeout defaults

Configuration changes should be versioned and rollout behavior understood.

## 12. Secret

Kubernetes Secret is a mechanism for distributing sensitive values to Pods, but base64 encoding itself is not encryption.

Use:
- encryption at rest where available
- external secret managers where appropriate
- RBAC least privilege
- short-lived credentials

## 13. Probes

### startupProbe
Has the slow-loading model/runtime completed startup?

### readinessProbe
Can this Pod receive traffic now?

### livenessProbe
Is the process stuck enough that restart may help?

Do not make probes run expensive generation requests.

## 14. GPU Scheduling

GPU nodes need vendor drivers and a device plugin. GPUs are exposed as integer extended resources and are not overcommitted like ordinary CPU.

Example concept:

~~~yaml
resources:
  limits:
    nvidia.com/gpu: 1
~~~

Node labels/affinity can select GPU type or memory class.

## 15. Rolling Update

For desired replicas R:

~~~text
max total Pods during rollout
≈ R + maxSurge

minimum available
≈ R - maxUnavailable
~~~

Percentages are resolved by Kubernetes according to rollout rules; capacity planning must include the transient surge.

## 16. Rollout Safety

Before promoting a new model/image:
- startup succeeds
- readiness passes
- error rate acceptable
- latency/SLO acceptable
- model-quality gate passes

Deployment health alone does not prove model quality.

## 17. HPA

HorizontalPodAutoscaler adjusts replica count based on observed metrics.

Current stable API:

~~~yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
~~~

For AI serving, CPU alone is often a weak scaling signal. Consider queue depth, running requests, tokens/sec, latency or other custom/external metrics.

## 18. HPA Control Loop

Conceptually for a metric:

~~~text
desiredReplicas
≈
ceil(
currentReplicas
× currentMetric / targetMetric
)
~~~

Real HPA behavior also includes readiness handling, tolerance and stabilization.

## 19. Scale-to-Zero

Scale-to-zero has special metric/runtime requirements and adds cold-start cost. GPU model-loading time can make aggressive scale-to-zero unsuitable for latency-sensitive services.

## 20. Cluster Autoscaling

HPA creates demand for Pods; node autoscaling creates/removes compute nodes.

For scarce GPUs, account for:
- provisioning delay
- device availability
- image/model download time
- quota

## 21. Canary

Route a small share of traffic to a candidate.

Compare:
- error rate
- TTFT / p99
- model-quality metrics
- resource use

Automatic rollback should use explicit gates.

## 22. Blue-Green

Run old and new stacks side by side and switch traffic.

Pros:
- fast rollback

Cons:
- temporarily doubles expensive capacity

## 23. Cloud Building Blocks

Most clouds provide equivalents of:
- image registry
- managed Kubernetes
- object storage
- load balancer
- secret manager
- monitoring/logging
- GPU node pools

Learn concepts before provider-specific buttons.

## 24. Stateful Dependencies

Model servers should generally avoid storing critical state on local ephemeral disks.

Externalize:
- model artifacts
- experiment metadata
- datasets
- vector indexes where required

## 25. Graceful Termination

On termination:
1. mark unready
2. stop new traffic
3. drain/cancel active requests
4. flush telemetry
5. exit before termination grace period

## 26. Deployment Math

src/deployment_math.py implements:

- desired_replicas_from_metric
- rolling_update_bounds
- canary_request_counts
- availability
- capacity_with_headroom
- gpu_pool_capacity

## 27. Example Manifests

The chapter includes example Docker/Kubernetes artifacts. They use placeholder image names and no real credentials.

## 28. Common Mistakes

1. latest tag used as rollback identity
2. secret baked into image
3. liveness starts before model has loaded
4. readiness checks process instead of serving capability
5. GPU limit omitted
6. HPA based only on misleading CPU utilization
7. rollout surge exceeds GPU quota
8. image/model download time ignored
9. deployment health mistaken for model quality
10. no rollback trigger

## 29. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 30. Checklist

- [ ] API contract
- [ ] immutable container
- [ ] requests/limits
- [ ] probes
- [ ] Deployment / Service
- [ ] GPU resource
- [ ] ConfigMap / Secret
- [ ] Gateway / Ingress
- [ ] HPA
- [ ] rolling/canary
- [ ] cloud building blocks
- [ ] rollback

## 31. What's Next

Chapter 65 manages experiments, artifacts, model promotion and production monitoring across the ML lifecycle.