# Chapter 42 — Speech: Audio, STT & TTS

## 1. Speech Is a Time-Domain Signal

Raw digital audio is a sequence of sampled amplitudes:

~~~text
x[0], x[1], ..., x[N-1]
~~~

At sample rate f_s:

~~~text
duration_seconds =
N / f_s
~~~

Speech systems often transform this waveform before modeling.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain sample rate and waveform
- explain aliasing/Nyquist intuition
- frame audio signals
- derive the STFT concept
- compute magnitude/power spectra
- explain mel frequency scaling
- build log-mel spectrograms
- explain STT pipelines
- explain CTC and blank tokens
- perform greedy CTC decoding
- explain encoder-decoder speech recognition
- understand Whisper-style seq2seq concepts
- explain TTS acoustic models
- explain vocoders
- distinguish autoregressive vs non-autoregressive TTS
- understand speech evaluation metrics

## 3. Sample Rate

Sample rate:

~~~text
samples per second
~~~

Common examples:
- 16 kHz for many speech-recognition pipelines
- higher rates for general audio/music

A 1-second waveform at 16 kHz has 16,000 samples.

## 4. Nyquist Intuition

To represent a frequency reliably, the sample rate must be sufficiently high.

Nyquist frequency:

~~~text
f_Nyquist =
f_s / 2
~~~

Frequencies above this can alias into lower frequencies if not filtered before sampling/downsampling.

## 5. Amplitude Normalization

Audio may be stored as:
- integer PCM
- floating point

Neural pipelines commonly convert to floating-point ranges such as approximately [-1,1].

Never assume every file uses identical gain or loudness.

## 6. Framing

Speech changes over time.

Instead of one Fourier transform over an entire recording, split waveform into overlapping frames:

~~~text
frame length
hop length
~~~

Example:
- 25 ms frame
- 10 ms hop

Exact choices vary.

## 7. Window Function

Multiplying each frame by a tapered window reduces spectral leakage.

Common:

~~~text
Hann window
~~~

The window is applied before the FFT.

## 8. STFT

Short-Time Fourier Transform:

~~~text
waveform
↓ frame
↓ window
↓ FFT each frame
↓
complex time-frequency matrix
~~~

For real signals, one-sided FFT keeps non-redundant frequencies.

## 9. Complex STFT

STFT contains:
- magnitude
- phase

~~~text
X(t,f)
=
real + j imaginary
~~~

Magnitude:

~~~text
|X|
~~~

Power:

~~~text
|X|^2
~~~

## 10. Spectrogram

A spectrogram visualizes energy over:

~~~text
time × frequency
~~~

Speech formants, harmonics and transients become visible.

## 11. Mel Scale

Human frequency perception is not linear.

A common conversion:

~~~text
mel(f)
=
2595 log10(
1 + f/700
)
~~~

Mel filter banks compress linear-frequency bins into perceptually motivated bands.

## 12. Mel Filter Bank

Construct triangular filters across FFT frequencies.

Then:

~~~text
mel_power
=
power_spectrogram
× mel_filterbank
~~~

depending on matrix orientation.

## 13. Log-Mel Spectrogram

Apply log compression:

~~~text
log_mel =
log(mel_power + epsilon)
~~~

This reduces huge dynamic ranges.

Many speech models operate on log-mel-like features.

## 14. STT Pipeline

Classic conceptual pipeline:

~~~text
waveform
↓
features
↓
acoustic / encoder model
↓
token probabilities
↓
decoder
↓
text
~~~

Modern end-to-end systems learn much more of this pipeline jointly.

## 15. CTC

Connectionist Temporal Classification allows training when:
- input frame sequence is longer
- target token sequence is shorter
- exact frame-to-token alignment is unknown

CTC introduces a blank symbol.

## 16. CTC Paths

Suppose target:

~~~text
A B
~~~

Possible frame path:

~~~text
blank A A blank B
~~~

Collapse:
1. merge repeated adjacent labels
2. remove blanks

Result:

~~~text
A B
~~~

## 17. Repeated Token Caveat

To output:

~~~text
A A
~~~

a CTC path requires separation, e.g.:

~~~text
A blank A
~~~

Otherwise adjacent repeated A labels collapse to one.

## 18. CTC Loss Inputs

Framework CTC commonly needs:
- log probabilities
- targets
- input lengths
- target lengths
- blank id

Input time must be long enough to align to targets.

## 19. CTC Greedy Decode

Per frame:

~~~text
argmax class
~~~

Then collapse repeats and remove blank.

Greedy decoding is simple but not always globally optimal.

## 20. CTC Beam Search

Beam search can combine:
- acoustic probabilities
- language-model score
- token constraints

This can outperform greedy decoding.

## 21. Encoder-Decoder STT

Alternative:

~~~text
audio encoder
↓
audio memory
↓
autoregressive text decoder
↓
transcript
~~~

This resembles Chapter 36 seq2seq models.

## 22. Whisper-Style Concepts

A modern seq2seq speech recognizer can use:
- log-mel spectrogram input
- Transformer encoder
- autoregressive Transformer decoder
- language/task/timestamp tokens

The exact tokenizer, audio window and decoding protocol are model-specific.

## 23. Streaming STT

Real-time transcription introduces:
- chunking
- limited future context
- latency/accuracy trade-offs
- state/cache management

A model designed for offline full-context transcription is not automatically streaming.

## 24. TTS Pipeline

Text-to-speech:

~~~text
text / phonemes
↓
text encoder
↓
acoustic model
↓
mel / acoustic representation
↓
vocoder
↓
waveform
~~~

Some modern systems instead generate waveform/audio tokens more directly.

## 25. Text Front End

TTS may use:
- graphemes
- phonemes
- subword/text tokens

Pronunciation modeling matters for:
- names
- numbers
- multilingual text
- ambiguous spelling

## 26. Acoustic Model

Maps linguistic input to an acoustic representation such as mel spectrogram.

Must learn:
- duration
- pitch/prosody
- energy
- pronunciation

## 27. Vocoder

Converts acoustic representation to waveform.

Families include:
- autoregressive neural vocoders
- GAN vocoders
- flow-based vocoders
- diffusion/score-based vocoders

The vocoder strongly affects audio quality and speed.

## 28. Autoregressive TTS

Generates output step-by-step.

Pros:
- expressive sequential modeling

Cons:
- slower
- exposure/attention failures can occur

## 29. Non-Autoregressive TTS

Predicts many output frames in parallel, often with explicit duration modeling.

Pros:
- faster synthesis

Challenges:
- duration/prosody modeling

## 30. Speech Tokens / Codec Models

Modern audio systems may compress waveform into discrete or continuous learned codec latents.

Then a Transformer can model audio tokens similarly to language tokens.

This connects speech with multimodal and generative-model chapters.

## 31. STT Metrics

Word Error Rate:

~~~text
WER =
(S + D + I) / N
~~~

where:
- S substitutions
- D deletions
- I insertions
- N reference words

Character Error Rate uses characters instead of words.

## 32. TTS Evaluation

Possible metrics:
- intelligibility
- speaker similarity
- pitch/duration errors
- human Mean Opinion Score
- task-specific listening tests

Automated metrics do not fully capture naturalness.

## 33. Voice Identity & Consent

Speech generation can imitate vocal characteristics.

Real deployments should use:
- consent
- disclosure where appropriate
- provenance/watermarking policies where applicable
- safeguards against impersonation/deception

This course focuses on general TTS mechanics, not impersonating real people.

## 34. PyTorch STFT API

For real waveform input, current PyTorch expects explicit complex-output handling.

Use:

~~~python
window = torch.hann_window(n_fft)

spec = torch.stft(
    waveform,
    n_fft=n_fft,
    hop_length=hop,
    window=window,
    return_complex=True,
)
~~~

Then:

~~~python
power = spec.abs().pow(2)
~~~

## 35. From Scratch

src/speech_numpy.py includes:

- frame_signal
- stft
- hz_to_mel
- mel_to_hz
- mel_filterbank
- log_mel_spectrogram
- ctc_greedy_decode
- word_error_rate

## 36. Common Mistakes

1. sample-rate mismatch
2. no anti-aliasing before naive downsampling
3. inconsistent waveform normalization
4. forgetting window before STFT
5. mixing magnitude and power spectra
6. wrong mel filter dimensions
7. log(0) without epsilon
8. CTC blank included in target labels
9. collapsing CTC repeats in wrong order
10. reporting WER with inconsistent text normalization

## 37. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 38. Checklist

- [ ] sample rate
- [ ] frames / hop
- [ ] Hann window
- [ ] STFT
- [ ] magnitude / power
- [ ] mel scale
- [ ] log-mel
- [ ] CTC
- [ ] CTC decode
- [ ] seq2seq STT
- [ ] TTS acoustic model
- [ ] vocoder
- [ ] WER/CER

## 39. What's Next

Batch 15 moves into recommender systems, object detection/segmentation and embedding/vector databases.
