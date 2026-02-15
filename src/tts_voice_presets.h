/*
 * tts_voice_presets.h - Precomputed speaker embeddings for voice cloning
 *
 * Binary file format:
 *   [n:int32] count of presets
 *   n x [name:64 bytes (null-padded), embed:1024 x float32]
 *
 * File location: <tts-model-dir>/voice_presets.bin
 * Missing file is not an error (just no presets available).
 */

#ifndef LOCAL_AI_TTS_VOICE_PRESETS_H
#define LOCAL_AI_TTS_VOICE_PRESETS_H

#define TTS_PRESET_NAME_LEN     64
#define TTS_PRESET_EMBED_DIM    1024

typedef struct {
    char name[TTS_PRESET_NAME_LEN];
    float embed[TTS_PRESET_EMBED_DIM];
} tts_voice_preset_t;

typedef struct {
    tts_voice_preset_t *presets;
    int n_presets;
} tts_voice_presets_t;

/* Load voice presets from binary file.
 * Returns 0 on success, -1 if file not found or invalid.
 * Missing file is expected (not an error). */
int tts_voice_presets_load(tts_voice_presets_t *vp, const char *path);

/* Free loaded presets. */
void tts_voice_presets_free(tts_voice_presets_t *vp);

/* Find a preset by name (case-insensitive).
 * Returns pointer to 1024-dim embedding, or NULL if not found. */
const float *tts_voice_presets_find(const tts_voice_presets_t *vp, const char *name);

#endif /* LOCAL_AI_TTS_VOICE_PRESETS_H */
