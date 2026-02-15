/*
 * tts_mel.h - 128-band mel spectrogram for speaker encoder input
 *
 * Parameters match upstream extract_speaker_embedding():
 *   sample_rate=24000, n_fft=1024, hop=256, win=1024
 *   n_mels=128, fmin=0, fmax=12000
 *   Hann window, slaney-norm mel filterbank
 *   Reflect padding: 384 samples each side
 *   Log compression: log(clamp(x, min=1e-5))
 */

#ifndef LOCAL_AI_TTS_MEL_H
#define LOCAL_AI_TTS_MEL_H

#define TTS_MEL_N_FFT       1024
#define TTS_MEL_HOP         256
#define TTS_MEL_WIN         1024
#define TTS_MEL_N_MELS      128
#define TTS_MEL_FMIN        0.0f
#define TTS_MEL_FMAX        12000.0f
#define TTS_MEL_SAMPLE_RATE 24000
#define TTS_MEL_PAD         384  /* reflect pad on each side */
#define TTS_MEL_N_BINS      (TTS_MEL_N_FFT / 2 + 1)  /* 513 */

/* Compute 128-band log-mel spectrogram from 24kHz audio.
 * Input:  audio[n_samples] -- float32 PCM samples
 * Output: mel[128 * n_frames] -- row-major [n_mels, n_frames]
 *         n_frames = (n_samples + 2 * TTS_MEL_PAD) / TTS_MEL_HOP
 *
 * Returns malloc'd mel buffer, sets *out_n_frames.
 * Returns NULL on error. Caller must free(). */
float *tts_mel_compute(const float *audio, int n_samples, int *out_n_frames);

#endif /* LOCAL_AI_TTS_MEL_H */
