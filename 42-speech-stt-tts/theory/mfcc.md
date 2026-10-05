# MFCC — Mel-Frequency Cepstral Coefficients

MFCCs are compact speech/audio features derived from short-time spectra.

Pipeline:
1. frame the waveform into short overlapping windows
2. apply a window function
3. compute FFT / power spectrum
4. apply a Mel-spaced filter bank
5. take log filter-bank energies
6. apply a discrete cosine transform (DCT)
7. keep the lower-order cepstral coefficients

The Mel mapping is commonly written as:

m = 2595 log10(1 + f/700)

MFCCs do not preserve the full waveform. They compress spectral-envelope information that is useful for many classical speech tasks.

## Important choices
- sample rate
- frame length and hop
- FFT size
- number of Mel filters
- number of MFCC coefficients
- optional energy, delta and delta-delta features
- normalization

## Debug checks
Verify sample rate first, then inspect waveform, spectrum, Mel energies and final coefficient ranges. Compare shapes for a fixed one-second clip.

## Modern context
End-to-end neural speech systems often learn directly from waveform or log-Mel features, but MFCC remains important for understanding classical ASR, compact features and signal-processing foundations.
