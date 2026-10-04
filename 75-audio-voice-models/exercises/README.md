# Chapter 75 Exercises

1. Convert seconds to sample counts at 16k/24k/48kHz.
2. Build overlapping streaming frame ranges.
3. Compute RMS and energy-based VAD flags.
4. Tune endpoint silence and measure turn-end latency.
5. Calculate codec bitrate for multiple codebook configurations.
6. Calculate audio tokens/sec.
7. Compare RTF with first-response latency.
8. Design a barge-in cancellation state machine.
9. Calculate jitter-buffer milliseconds.
10. Design evaluation slices for noise/accent/device/overlap.

Challenge:
- build a local streaming VAD + endpoint simulator
- connect mock streaming ASR -> text responder -> mock streaming TTS with cancellation
