"""Verify mel spectrogram implementation against independent numpy reference.

Two independent implementations of the same spec (slaney mel, 128 bands,
24kHz, reflect padding) are compared. One mirrors tts_mel.c loop-by-loop,
the other uses numpy idioms as a cross-check. Also validates known-frequency
signals land in the correct mel bins.

Fully headless -- no server, no pytorch, only numpy.

Usage:
    python tools/verify_mel.py              # synthetic test signal
    python tools/verify_mel.py audio.wav    # from WAV file
"""
import numpy as np
import sys
import struct
import os

# Parameters matching tts_mel.h
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 128
FMIN = 0.0
FMAX = 12000.0
SAMPLE_RATE = 24000
PAD = 384
N_BINS = N_FFT // 2 + 1  # 513


# ========================================================================
# Implementation A: mirrors tts_mel.c loop-by-loop
# ========================================================================

def hz_to_mel_slaney(hz):
    if hz < 1000.0:
        return hz * 3.0 / 200.0
    return 15.0 + 27.0 * np.log(hz / 1000.0) / np.log(6.4)

def mel_to_hz_slaney(mel):
    if mel < 15.0:
        return mel * 200.0 / 3.0
    return 1000.0 * np.exp((mel - 15.0) * np.log(6.4) / 27.0)

def build_filterbank_A():
    """Build mel filterbank matching tts_mel.c scalar loops."""
    mel_min = hz_to_mel_slaney(FMIN)
    mel_max = hz_to_mel_slaney(FMAX)
    n_edges = N_MELS + 2
    mel_edges = [mel_min + (mel_max - mel_min) * i / (n_edges - 1) for i in range(n_edges)]
    hz_edges = [mel_to_hz_slaney(m) for m in mel_edges]
    fft_freqs = [float(SAMPLE_RATE) * i / N_FFT for i in range(N_BINS)]

    fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    for m in range(N_MELS):
        left, center, right = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        norm = 2.0 / (right - left)
        for k in range(N_BINS):
            f = fft_freqs[k]
            val = 0.0
            if f >= left and f <= center and center > left:
                val = (f - left) / (center - left)
            elif f > center and f <= right and right > center:
                val = (right - f) / (right - center)
            fb[m, k] = val * norm
    return fb

def mel_spectrogram_A(audio):
    """C-matching: scalar reflect pad, scalar Hann, per-frame FFT."""
    n = len(audio)
    padded_len = n + 2 * PAD
    padded = np.zeros(padded_len, dtype=np.float32)
    for i in range(PAD):
        src = PAD - i
        if src >= n: src = n - 1
        padded[i] = audio[src]
    padded[PAD:PAD + n] = audio
    for i in range(PAD):
        src = n - 2 - i
        if src < 0: src = 0
        padded[PAD + n + i] = audio[src]

    hann = np.array([0.5 * (1.0 - np.cos(2.0 * np.pi * i / WIN))
                      for i in range(WIN)], dtype=np.float32)
    n_frames = (padded_len - N_FFT) // HOP + 1
    fb = build_filterbank_A()
    mel = np.zeros((N_MELS, n_frames), dtype=np.float32)

    for t in range(n_frames):
        frame = padded[t * HOP:t * HOP + N_FFT] * hann
        spectrum = np.fft.rfft(frame)
        mag = np.sqrt(spectrum.real**2 + spectrum.imag**2 + 1e-9).astype(np.float32)
        mel_frame = fb @ mag
        mel[:, t] = np.log(np.maximum(mel_frame, 1e-5))

    return mel, n_frames


# ========================================================================
# Implementation B: independent numpy-idiomatic reference
# ========================================================================

def _slaney_hz2mel(f):
    """Vectorized slaney Hz->mel (librosa formula)."""
    f = np.asarray(f, dtype=np.float64)
    with np.errstate(divide='ignore'):
        out = np.where(f < 1000.0, f * 3.0 / 200.0,
                       15.0 + 27.0 * np.log(np.maximum(f, 1e-30) / 1000.0) / np.log(6.4))
    return out

def _slaney_mel2hz(m):
    """Vectorized slaney mel->Hz."""
    m = np.asarray(m, dtype=np.float64)
    out = np.where(m < 15.0, m * 200.0 / 3.0,
                   1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0))
    return out

def build_filterbank_B():
    """Build mel filterbank using vectorized numpy (independent impl)."""
    mel_pts = np.linspace(_slaney_hz2mel(FMIN), _slaney_hz2mel(FMAX), N_MELS + 2)
    hz_pts = _slaney_mel2hz(mel_pts)
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_BINS)

    fb = np.zeros((N_MELS, N_BINS), dtype=np.float64)
    for m in range(N_MELS):
        lo, mid, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        # Rising slope
        up = (fft_freqs - lo) / (mid - lo)
        # Falling slope
        down = (hi - fft_freqs) / (hi - mid)
        fb[m] = np.maximum(0, np.minimum(up, down))
        # Slaney area normalization
        fb[m] *= 2.0 / (hi - lo)
    return fb.astype(np.float32)

def mel_spectrogram_B(audio):
    """Numpy-idiomatic: np.pad reflect, vectorized filterbank."""
    padded = np.pad(audio, (PAD, PAD), mode='reflect')
    hann = np.hanning(WIN + 1)[:WIN].astype(np.float32)  # periodic Hann, same as 0.5*(1-cos(2pi*i/N))

    n_frames = (len(padded) - N_FFT) // HOP + 1
    fb = build_filterbank_B()
    mel = np.zeros((N_MELS, n_frames), dtype=np.float32)

    for t in range(n_frames):
        frame = padded[t * HOP:t * HOP + N_FFT] * hann
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum).astype(np.float32)
        # No +1e-9 in magnitude (reference uses clean abs)
        mel_frame = fb @ mag
        mel[:, t] = np.log(np.maximum(mel_frame, 1e-5))

    return mel, n_frames


# ========================================================================
# Frequency bin validation
# ========================================================================

def test_frequency_bins():
    """Verify that pure tones land in the correct mel bins."""
    print("Test: pure tone -> mel bin mapping")
    fb = build_filterbank_A()
    ok = True

    test_freqs = [200, 500, 1000, 2000, 4000, 8000]
    for freq in test_freqs:
        # Generate 0.1s pure tone
        n = SAMPLE_RATE // 10
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        mel, nf = mel_spectrogram_A(tone)
        # Find the mel bin with highest average energy
        avg_energy = mel.mean(axis=1)
        peak_bin = np.argmax(avg_energy)
        peak_val = avg_energy[peak_bin]

        # Expected bin: find which filter has max response at this frequency
        fft_bin = int(round(freq * N_FFT / SAMPLE_RATE))
        expected_bin = np.argmax(fb[:, fft_bin])

        match = abs(peak_bin - expected_bin) <= 1
        status = "ok" if match else "FAIL"
        if not match:
            ok = False
        print(f"  {freq:5d} Hz: peak_mel_bin={peak_bin:3d}, expected~{expected_bin:3d} [{status}]")

    return ok


# ========================================================================
# Helpers
# ========================================================================

def load_wav(path):
    """Load WAV file, return float32 samples."""
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a WAV file")
        f.read(4)
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a WAV file")
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                n_channels = struct.unpack("<H", fmt_data[2:4])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                break
            else:
                f.read(chunk_size)
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"data":
                raw = f.read(chunk_size)
                break
            else:
                f.read(chunk_size)

    if audio_format != 1:
        raise ValueError(f"Unsupported audio format: {audio_format}")
    if bits_per_sample == 16:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

    if n_channels > 1:
        samples = samples[::n_channels]

    print(f"  Loaded WAV: {sample_rate} Hz, {len(samples)} samples, "
          f"{len(samples)/sample_rate:.2f}s")
    if sample_rate != SAMPLE_RATE:
        print(f"  WARNING: sample rate {sample_rate} != expected {SAMPLE_RATE}")
    return samples


def generate_test_signal(duration_s=2.0, seed=42):
    """Deterministic multi-frequency test signal."""
    rng = np.random.RandomState(seed)
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE
    signal = np.zeros(n_samples, dtype=np.float32)
    for f in [100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000]:
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.15)
        signal += (amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)
    signal += (0.01 * rng.randn(n_samples)).astype(np.float32)
    return signal


def compare(name_a, mel_a, name_b, mel_b, threshold=0.9999):
    """Compare two mel spectrograms."""
    if mel_a.shape != mel_b.shape:
        print(f"  SHAPE MISMATCH: {name_a}={mel_a.shape} vs {name_b}={mel_b.shape}")
        nf = min(mel_a.shape[1], mel_b.shape[1])
        mel_a, mel_b = mel_a[:, :nf], mel_b[:, :nf]

    diff = np.abs(mel_a - mel_b)
    max_diff = diff.max()
    rms_diff = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(mel_a.flatten(), mel_b.flatten())[0, 1]

    ok = corr > threshold
    status = "PASS" if ok else "FAIL"
    print(f"  {name_a} vs {name_b}: {status}")
    print(f"    corr={corr:.8f}, max_diff={max_diff:.6f}, rms_diff={rms_diff:.6f}")

    if not ok:
        frame_err = np.sqrt(np.mean(diff ** 2, axis=0))
        worst_t = np.argmax(frame_err)
        print(f"    worst frame t={worst_t}: frame_rms={frame_err[worst_t]:.6f}")
        print(f"    {name_a}[:,{worst_t}][:5] = {mel_a[:5, worst_t]}")
        print(f"    {name_b}[:,{worst_t}][:5] = {mel_b[:5, worst_t]}")
    return ok


# ========================================================================
# Main
# ========================================================================

def main():
    all_ok = True

    # Load or generate audio
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        print(f"Loading audio from {sys.argv[1]}...")
        audio = load_wav(sys.argv[1])
    else:
        print("Generating deterministic test signal (2s, seed=42)...")
        audio = generate_test_signal()
        print(f"  {len(audio)} samples, range=[{audio.min():.4f}, {audio.max():.4f}]")

    # 1. Filterbank comparison
    print("\n--- Filterbank comparison ---")
    fb_a = build_filterbank_A()
    fb_b = build_filterbank_B()
    fb_diff = np.abs(fb_a - fb_b)
    fb_max = fb_diff.max()
    fb_corr = np.corrcoef(fb_a.flatten(), fb_b.flatten())[0, 1]
    fb_ok = fb_max < 1e-4
    print(f"  Filterbank A vs B: {'PASS' if fb_ok else 'FAIL'}")
    print(f"    max_diff={fb_max:.8f}, corr={fb_corr:.8f}")
    print(f"    shape={fb_a.shape}, nonzero={np.count_nonzero(fb_a)}")
    all_ok = all_ok and fb_ok

    # 2. Full mel comparison
    print("\n--- Mel spectrogram: impl A (C-matching) ---")
    mel_a, nf_a = mel_spectrogram_A(audio)
    print(f"  shape=[{N_MELS}, {nf_a}], range=[{mel_a.min():.3f}, {mel_a.max():.3f}]")

    print("\n--- Mel spectrogram: impl B (numpy reference) ---")
    mel_b, nf_b = mel_spectrogram_B(audio)
    print(f"  shape=[{N_MELS}, {nf_b}], range=[{mel_b.min():.3f}, {mel_b.max():.3f}]")

    print("\n--- Cross-check ---")
    ok = compare("impl_A", mel_a, "impl_B", mel_b)
    all_ok = all_ok and ok

    # 3. Frequency bin test
    print("\n--- Frequency bin validation ---")
    ok = test_frequency_bins()
    all_ok = all_ok and ok

    # 4. C dump comparison (if available)
    c_dump = "mel_out.raw"
    if os.path.exists(c_dump):
        print(f"\n--- C dump comparison ({c_dump}) ---")
        mel_dump = np.fromfile(c_dump, dtype=np.float32).reshape(N_MELS, -1)
        print(f"  shape=[{mel_dump.shape[0]}, {mel_dump.shape[1]}]")
        ok = compare("impl_A", mel_a, "C-dump", mel_dump)
        all_ok = all_ok and ok

    # Summary
    if all_ok:
        print("\nAll checks PASSED")
    else:
        print("\nSome checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
