# Batch 25 Integration Project — Reasoning, VLM & Real-Time Voice Lab

Batch 25 connects three interactive AI capabilities:

~~~text
budgeted reasoning
↓
visual grounding / multimodal context
↓
streaming voice interface
~~~

## Part A — Reasoning

Compare:
- majority/self-consistency
- best-of-N verifier selection
- pass@k
- token-budget allocation
- solve-per-token efficiency

## Part B — VLM

Calculate:
- image patch/token counts
- multi-image context usage
- remaining output/text budget
- grounding IoU/precision/recall

## Part C — Voice

Simulate:
- VAD
- endpointing
- neural-codec bitrate
- real-time factor
- first-response latency budget

## Full Gate

The combined system passes only when:
- reasoning selection fits its compute budget
- multimodal context fits and grounding succeeds
- streaming voice has an endpoint, RTF below 1, and sub-second modeled first response

## Run

~~~bash
python integration-project-batch25/src/reasoning_vlm_voice_lab.py --mode reasoning
python integration-project-batch25/src/reasoning_vlm_voice_lab.py --mode vlm
python integration-project-batch25/src/reasoning_vlm_voice_lab.py --mode voice
python integration-project-batch25/src/reasoning_vlm_voice_lab.py --mode full
~~~

## PyTorch Framework Smoke

CI also validates:
- patch count with torch.nn.Unfold
- NumPy visual projector against torch.nn.Linear
- NumPy RMS against PyTorch tensor math

No large multimodal/audio model is downloaded in CI.

## Required Extensions

1. verifier-guided search on a symbolic task
2. compute-matched self-consistency benchmark
3. tiny trainable image-patch projector
4. multi-image context-budget stress test
5. OCR/grounding evaluation slices
6. streaming VAD false-positive benchmark
7. barge-in cancellation simulator
8. partial-ASR revision-rate metric
9. optional supported Transformers multimodal/audio processor lab
10. final accuracy-latency-token Pareto report