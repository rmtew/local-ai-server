/*
 * tts_voice_presets.h - Precomputed speaker embeddings for voice cloning
 *
 * Binary file format:
 *   [n:int32] count of presets
 *   n x [name:64 bytes (null-padded), embed:embed_dim x float32]
 *
 * embed_dim is auto-detected from file size: 1024 (0.6B) or 2048 (1.7B).
 * File location: <tts-model-dir>/voice_presets.bin
 * Missing file is not an error (just no presets available).
 */

#ifndef LOCAL_AI_TTS_VOICE_PRESETS_H
#define LOCAL_AI_TTS_VOICE_PRESETS_H

#define TTS_PRESET_NAME_LEN     64

typedef struct {
    char *names;        /* [n_presets * TTS_PRESET_NAME_LEN], null-padded */
    float *embeds;      /* [n_presets * embed_dim] */
    int n_presets;
    int embed_dim;      /* 1024 or 2048, detected from file */
} tts_voice_presets_t;

/* Load voice presets from binary file.
 * Returns 0 on success, -1 if file not found or invalid.
 * Missing file is expected (not an error). */
int tts_voice_presets_load(tts_voice_presets_t *vp, const char *path);

/* Free loaded presets. */
void tts_voice_presets_free(tts_voice_presets_t *vp);

/* Find a preset by name (case-insensitive).
 * Returns pointer to embed_dim-dim embedding, or NULL if not found. */
const float *tts_voice_presets_find(const tts_voice_presets_t *vp, const char *name);

#endif /* LOCAL_AI_TTS_VOICE_PRESETS_H */
