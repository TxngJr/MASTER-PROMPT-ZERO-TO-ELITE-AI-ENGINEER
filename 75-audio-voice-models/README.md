# Chapter 75 — Audio & Voice Models

## 1. From Speech Pipelines to Audio-Language Systems

Traditional spoken assistant:

~~~text
audio
↓ ASR
text
↓ LLM
text
↓ TTS
audio
~~~

Modern audio-language systems may also encode audio directly into continuous/discrete representations and reason/generate across audio and text.

## 2. Learning Objectives

- distinguish waveform, features and audio tokens
- understand neural audio codec concepts
- calculate audio token rate / bitrate
- understand streaming ASR and streaming TTS
- implement simple energy-based VAD
- understand endpointing
- calculate real-time factor
- decompose first-response latency
- understand turn-taking and barge-in
- reason about full-duplex voice
- distinguish cascaded vs end-to-end speech systems
- understand audio-text-to-text models
- design voice evaluation slices

## 3. Waveform Recap

Digital audio is a sampled waveform.

~~~text
samples = duration_seconds * sample_rate
~~~

Chapter 42 covered waveform/STFT/mel/CTC foundations. This chapter focuses on real-time and generative audio systems.

## 4. Continuous Audio Features

An audio encoder can consume:
- waveform
- log-mel features
- learned frontend features

and produce continuous hidden representations.

## 5. Neural Audio Tokens

Neural codecs can compress audio into one or more sequences of discrete codebook IDs.

Conceptually:

~~~text
waveform
↓ encoder
latent
↓ vector quantization/codebooks
discrete audio tokens
↓ decoder
reconstructed waveform
~~~

Audio-language models can model some combination of semantic/acoustic/audio tokens.

## 6. Codec Bitrate

For C codebooks, codebook size K and F codec frames/sec:

~~~text
bits/code = ceil(log2(K))
bitrate = C * bits/code * F
~~~

This is an ideal raw-code bitrate and excludes metadata/container overhead.

## 7. Audio Token Rate

~~~text
tokens/sec = total_audio_tokens / duration_seconds
~~~

High token rates increase autoregressive generation cost.

## 8. Audio-Text-to-Text

Current Transformers documentation defines audio-text-to-text models as systems that accept audio plus text and generate text, enabling transcription, audio QA and instruction-following over audio. (see references.md)

This differs from classic ASR whose target is primarily transcription.

## 9. TTS

Current Transformers exposes text-to-audio/text-to-speech pipelines for supported speech synthesis models. (see references.md)

TTS evaluation includes intelligibility, naturalness, speaker/style consistency where appropriate, and latency.

## 10. Cascaded Voice System

~~~text
streaming ASR
↓ partial text
LLM
↓ partial response
streaming TTS
↓ audio chunks
speaker
~~~

Advantages:
- modular components
- easy text inspection

Costs:
- error propagation
- multiple latency stages

## 11. End-to-End Audio Models

An end-to-end system can consume speech/audio representations and directly predict text or audio representations.

Potential advantages:
- preserve paralinguistic cues
- fewer explicit stage boundaries

Challenges:
- training complexity
- controllability
- debugging/evaluation

## 12. Streaming ASR

Streaming ASR processes chunks before the utterance ends.

Outputs may be:
- unstable partial hypothesis
- stabilized partial text
- final transcript

Measure revision rate as well as final WER/CER.

## 13. Voice Activity Detection

VAD estimates whether a frame contains speech.

A simple educational energy VAD:

~~~text
speech = RMS(frame) >= threshold
~~~

Real systems use more robust learned/statistical methods because background noise breaks fixed energy thresholds.

## 14. Endpointing

Endpointing decides when the user's turn has ended.

One simple rule:

~~~text
after speech,
N consecutive silent frames
→ endpoint
~~~

Too short:
- cuts off pauses

Too long:
- sluggish assistant

## 15. Turn Taking

Conversation timing depends on:
- speech onset
- pauses
- endpoint confidence
- assistant preparation
- interruption

Good turn-taking is not just ASR accuracy.

## 16. Barge-In

Barge-in means the user starts speaking while assistant audio is playing.

A responsive system may:
1. detect new user speech
2. stop/fade assistant playback
3. cancel or reprioritize generation
4. resume listening

Cancellation semantics matter across LLM and TTS stages.

## 17. Full Duplex

Full-duplex systems listen and speak concurrently.

Additional challenges:
- echo
- overlapping speech
- interruption handling
- feedback loops
- concurrency/state synchronization

## 18. Echo Cancellation

If microphone hears the assistant's own speaker output, the system may transcribe/respond to itself.

Acoustic echo cancellation uses the known playback reference to suppress echo.

Treat this as a signal-processing subsystem, not a prompt-engineering problem.

## 19. Jitter / Buffers

Networks deliver audio chunks with variable delay.

Buffers trade:
- smooth playback
- additional latency

Track buffered milliseconds explicitly.

## 20. Real-Time Factor

~~~text
RTF = processing_time / audio_duration
~~~

RTF < 1 means processing is faster than audio duration under that benchmark.

Streaming UX still depends on chunking and first-result latency.

## 21. First-Response Latency

One useful decomposition:

~~~text
capture/chunk
+ endpoint or partial-ASR delay
+ model first token
+ TTS first audio
+ network
~~~

Optimize the slowest stage rather than one aggregate number only.

## 22. Streaming TTS

Instead of waiting for the full response text/audio, produce playable speech chunks early.

Challenges:
- prosody across future text
- sentence boundaries
- revisions
- cancellation

## 23. Voice Identity & Consent

Voice systems can imitate speaker characteristics.

Production design should include authorization/consent, disclosure where appropriate, and misuse controls rather than assuming speaker similarity is harmless.

## 24. Evaluation

STT/audio understanding:
- WER/CER
- audio QA accuracy
- robustness to noise/accent/device

TTS/voice:
- intelligibility
- naturalness
- pronunciation
- latency/RTF

Conversation:
- first-response latency
- interruption success
- turn-end latency
- false VAD activation

## 25. From Scratch

`src/voice.py` implements:
- sample_count
- frame_ranges
- rms_energy
- frame_rms
- vad_flags
- endpoint_after_silence
- real_time_factor
- codec_bitrate_kbps
- audio_tokens_per_second
- streaming_first_response_latency
- buffered_audio_ms

## 26. Common Mistakes

1. final WER used as the only streaming metric
2. partial transcript instability ignored
3. fixed VAD threshold assumed robust everywhere
4. endpoint silence too aggressive
5. RTF confused with first-response latency
6. audio token bitrate ignores codebook count
7. full duplex attempted without echo handling
8. barge-in fails to cancel downstream work
9. network jitter buffer omitted from latency
10. voice identity/consent ignored

## 27. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] audio tokens/codecs
- [ ] streaming ASR
- [ ] streaming TTS
- [ ] VAD
- [ ] endpointing
- [ ] turn taking
- [ ] barge-in
- [ ] full duplex
- [ ] RTF/latency
- [ ] voice evaluation

## 29. What's Next

Batch 26 covers world models, robotics AI and edge/TinyML systems.