# MFCC — Mel-Frequency Cepstral Coefficients

MFCCs are a compact speech feature representation built from short-time spectral information.

## Pipeline

~~~text
waveform
→ framing/windowing
→ FFT
→ power spectrum
→ Mel filterbank
→ log energies
→ DCT
→ selected cepstral coefficients
~~~

### 1. Frames

Speech is non-stationary globally but approximately stationary over short windows. Typical systems therefore analyze overlapping frames. Exact frame/window parameters are part of the experiment configuration.

### 2. Spectrum

For frame (x[n]), compute a discrete Fourier transform and power magnitude:

~~~text
P[k] ∝ |X[k]|^2
~~~

### 3. Mel scale

A common mapping is:

~~~text
mel(f) = 2595 * log10(1 + f/700)
~~~

Triangular filters aggregate spectral energy into perceptually motivated frequency bands.

### 4. Log

Use a positive floor before the logarithm:

~~~text
log(max(E_m, epsilon))
~~~

### 5. DCT

A discrete cosine transform decorrelates the log-Mel bands approximately; keep a configured number of low-order coefficients.

## MFCC vs log-Mel spectrogram

- MFCC: compact, historically strong for classical speech systems.
- log-Mel: preserves more local spectral structure and is common for modern neural speech models.

## Debugging invariants

- sample rate is known and consistent;
- frame count matches framing arithmetic;
- filterbank dimensions are correct;
- energies are floored before log;
- no NaN/Inf after silent frames;
- preprocessing parameters used in training and inference are identical.
