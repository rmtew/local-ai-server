"""
tts_pipeline_ffi.py - Python ctypes wrapper for tts_pipeline.dll

Provides a TtsPipeline class that loads the native TTS DLL and exposes
synthesis directly — no HTTP server needed.

Usage:
    from tts_pipeline_ffi import TtsPipeline

    with TtsPipeline(model_dir, fp16=True) as tts:
        result = tts.synthesize("Hello world", seed=42)
        # result.wav_data, result.n_steps, result.elapsed_ms, etc.
        print(tts.get_vram_mb())  # GPU VRAM in MB
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional


class SynthResult(NamedTuple):
    wav_data: bytes       # Complete WAV file (header + PCM)
    n_steps: int          # Autoregressive decode steps
    n_samples: int        # PCM sample count (24kHz)
    elapsed_ms: float     # Pipeline wall-clock time (from C)
    duration_s: float     # Audio duration = n_samples / 24000


def _find_dll() -> str:
    """Locate tts_pipeline.dll relative to this script."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if sys.platform == "win32":
        dll_name = "tts_pipeline.dll"
    else:
        dll_name = "tts_pipeline.so"

    dll_path = project_root / "bin" / dll_name
    if dll_path.exists():
        return str(dll_path)

    raise FileNotFoundError(
        f"TTS DLL not found at {dll_path}. Run: build.bat ttsdll"
    )


class TtsPipeline:
    """Native TTS pipeline via ctypes FFI."""

    def __init__(
        self,
        model_dir: str,
        fp16: bool = True,
        int8: bool = False,
        verbose: bool = False,
        threads: int = 4,
    ):
        self._dll = None
        self._handle = None

        dll_path = _find_dll()

        # On Windows, dependent DLLs (libopenblas, CUDA) must be findable.
        # Use winmode=0 to enable PATH-based DLL search, and prepend
        # bin/ and CUDA bin/ to PATH.
        if sys.platform == "win32":
            bin_dir = str(Path(dll_path).parent)
            extra_paths = [bin_dir]
            # Auto-detect CUDA bin for cublas/cudart DLLs
            cuda_dir = os.environ.get("CUDA_PATH", "")
            if not cuda_dir:
                # Try common locations
                for ver in ["v13.1", "v12.8", "v12.6"]:
                    candidate = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}"
                    if os.path.isdir(candidate):
                        cuda_dir = candidate
                        break
            if cuda_dir:
                cuda_bin = os.path.join(cuda_dir, "bin")
                if os.path.isdir(cuda_bin):
                    extra_paths.append(cuda_bin)
            os.environ["PATH"] = ";".join(extra_paths) + ";" + os.environ.get("PATH", "")
            self._dll = ctypes.CDLL(dll_path, winmode=0)
        else:
            self._dll = ctypes.CDLL(dll_path)
        self._setup_prototypes()

        # Create opaque pipeline handle
        self._handle = self._dll.tts_dll_create()
        if not self._handle:
            raise RuntimeError("tts_dll_create returned NULL")

        # Set threads before init (init may use them)
        self._dll.tts_dll_set_threads(threads)

        # Initialize pipeline
        rc = self._dll.tts_dll_init(
            self._handle,
            model_dir.encode("utf-8"),
            1 if fp16 else 0,
            1 if int8 else 0,
            1 if verbose else 0,
        )
        if rc != 0:
            self._dll.tts_dll_destroy(self._handle)
            self._handle = None
            raise RuntimeError(
                f"tts_dll_init failed (rc={rc}) for model: {model_dir}"
            )

    def _setup_prototypes(self):
        dll = self._dll

        dll.tts_dll_create.restype = ctypes.c_void_p
        dll.tts_dll_create.argtypes = []

        dll.tts_dll_destroy.restype = None
        dll.tts_dll_destroy.argtypes = [ctypes.c_void_p]

        dll.tts_dll_init.restype = ctypes.c_int
        dll.tts_dll_init.argtypes = [
            ctypes.c_void_p,    # tts
            ctypes.c_char_p,    # model_dir
            ctypes.c_int,       # fp16
            ctypes.c_int,       # int8
            ctypes.c_int,       # verbose
        ]

        dll.tts_dll_synthesize.restype = ctypes.c_int
        dll.tts_dll_synthesize.argtypes = [
            ctypes.c_void_p,                    # tts
            ctypes.c_char_p,                    # text
            ctypes.c_char_p,                    # voice
            ctypes.c_char_p,                    # language
            ctypes.c_float,                     # temperature
            ctypes.c_int,                       # top_k
            ctypes.c_int,                       # seed
            ctypes.POINTER(ctypes.c_int),       # n_steps
            ctypes.POINTER(ctypes.c_int),       # n_samples
            ctypes.POINTER(ctypes.c_double),    # elapsed_ms
            ctypes.POINTER(ctypes.c_void_p),    # wav_data
            ctypes.POINTER(ctypes.c_size_t),    # wav_len
        ]

        dll.tts_dll_free_wav.restype = None
        dll.tts_dll_free_wav.argtypes = [ctypes.c_void_p]

        dll.tts_dll_set_threads.restype = None
        dll.tts_dll_set_threads.argtypes = [ctypes.c_int]

        dll.tts_dll_get_vram_mb.restype = ctypes.c_int
        dll.tts_dll_get_vram_mb.argtypes = []

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.3,
        top_k: int = 50,
        seed: int = 42,
    ) -> SynthResult:
        """Synthesize speech. Returns SynthResult with WAV data and timing."""
        if not self._handle:
            raise RuntimeError("Pipeline not initialized")

        n_steps = ctypes.c_int(0)
        n_samples = ctypes.c_int(0)
        elapsed_ms = ctypes.c_double(0.0)
        wav_ptr = ctypes.c_void_p(0)
        wav_len = ctypes.c_size_t(0)

        rc = self._dll.tts_dll_synthesize(
            self._handle,
            text.encode("utf-8"),
            voice.encode("utf-8") if voice else None,
            language.encode("utf-8") if language else None,
            temperature,
            top_k,
            seed,
            ctypes.byref(n_steps),
            ctypes.byref(n_samples),
            ctypes.byref(elapsed_ms),
            ctypes.byref(wav_ptr),
            ctypes.byref(wav_len),
        )

        if rc != 0:
            raise RuntimeError(f"tts_dll_synthesize failed (rc={rc})")

        # Copy WAV data to Python bytes, then free C buffer
        wav_bytes = ctypes.string_at(wav_ptr.value, wav_len.value)
        self._dll.tts_dll_free_wav(wav_ptr)

        return SynthResult(
            wav_data=wav_bytes,
            n_steps=n_steps.value,
            n_samples=n_samples.value,
            elapsed_ms=elapsed_ms.value,
            duration_s=n_samples.value / 24000.0,
        )

    def get_vram_mb(self) -> int:
        """Return total GPU VRAM used (weights + buffers) in MB, or 0 if no GPU."""
        if self._dll:
            return self._dll.tts_dll_get_vram_mb()
        return 0

    def set_threads(self, n: int):
        """Set CPU thread count for vocoder."""
        if self._dll:
            self._dll.tts_dll_set_threads(n)

    def close(self):
        """Free pipeline resources."""
        if self._handle and self._dll:
            self._dll.tts_dll_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
