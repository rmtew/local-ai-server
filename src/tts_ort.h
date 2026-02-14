/*
 * tts_ort.h - ONNX Runtime context for Qwen3-TTS
 *
 * The full TTS pipeline (talker, code predictor, vocoder) is now native C.
 * ONNX Runtime is kept for potential future use (e.g. speaker_encoder
 * for voice cloning).
 */

#ifndef LOCAL_AI_TTS_ORT_H
#define LOCAL_AI_TTS_ORT_H

#ifdef USE_ORT

/* Forward-declare ORT types to avoid including the full header everywhere */
typedef struct OrtApi OrtApi;
typedef struct OrtEnv OrtEnv;
typedef struct OrtSession OrtSession;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtMemoryInfo OrtMemoryInfo;

/* ONNX Runtime context for Qwen3-TTS.
 * The full pipeline (talker, code predictor, vocoder) is now native C.
 * ONNX Runtime is kept for potential future use (e.g. speaker_encoder
 * for voice cloning). */
typedef struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSessionOptions *opts;
    OrtMemoryInfo *mem_info;

    OrtSession *speaker_encoder;     /* (future: voice cloning) */

    int verbose;
} TtsOrt;

/* Initialize ONNX Runtime and load optional sessions. Returns 0 on success. */
int tts_ort_init(TtsOrt *ort, const char *model_dir, int verbose);

/* Free all sessions and ORT resources. */
void tts_ort_free(TtsOrt *ort);

#endif /* USE_ORT */
#endif /* LOCAL_AI_TTS_ORT_H */
