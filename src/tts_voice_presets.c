/*
 * tts_voice_presets.c - Load and look up precomputed speaker embeddings
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_voice_presets.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

int tts_voice_presets_load(tts_voice_presets_t *vp, const char *path) {
    memset(vp, 0, sizeof(*vp));

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* Read count */
    int32_t n = 0;
    if (fread(&n, sizeof(int32_t), 1, f) != 1 || n <= 0 || n > 10000) {
        fclose(f);
        return -1;
    }

    /* Detect embed_dim from file size.
     * File = 4 (count) + n * (64 (name) + embed_dim * 4 (floats))
     * So: embed_dim = ((file_size - 4) / n - 64) / 4 */
    long data_size = file_size - 4;
    if (data_size <= 0 || data_size % n != 0) {
        fclose(f);
        return -1;
    }
    long entry_size = data_size / n;
    if (entry_size <= TTS_PRESET_NAME_LEN || (entry_size - TTS_PRESET_NAME_LEN) % sizeof(float) != 0) {
        fclose(f);
        return -1;
    }
    int embed_dim = (int)((entry_size - TTS_PRESET_NAME_LEN) / sizeof(float));
    if (embed_dim != 1024 && embed_dim != 2048) {
        fprintf(stderr, "voice_presets: unexpected embed_dim=%d (expected 1024 or 2048)\n", embed_dim);
        fclose(f);
        return -1;
    }

    /* Allocate storage */
    vp->names = (char *)calloc((size_t)n * TTS_PRESET_NAME_LEN, 1);
    vp->embeds = (float *)malloc((size_t)n * embed_dim * sizeof(float));
    if (!vp->names || !vp->embeds) {
        free(vp->names);
        free(vp->embeds);
        vp->names = NULL;
        vp->embeds = NULL;
        fclose(f);
        return -1;
    }

    /* Read entries */
    for (int i = 0; i < n; i++) {
        if (fread(vp->names + i * TTS_PRESET_NAME_LEN, 1, TTS_PRESET_NAME_LEN, f) != TTS_PRESET_NAME_LEN) {
            goto read_fail;
        }
        if (fread(vp->embeds + i * embed_dim, sizeof(float), embed_dim, f) != (size_t)embed_dim) {
            goto read_fail;
        }
    }

    fclose(f);
    vp->n_presets = n;
    vp->embed_dim = embed_dim;
    return 0;

read_fail:
    free(vp->names);
    free(vp->embeds);
    vp->names = NULL;
    vp->embeds = NULL;
    fclose(f);
    return -1;
}

void tts_voice_presets_free(tts_voice_presets_t *vp) {
    free(vp->names);
    free(vp->embeds);
    memset(vp, 0, sizeof(*vp));
}

const float *tts_voice_presets_find(const tts_voice_presets_t *vp, const char *name) {
    if (!vp->names || !name || name[0] == '\0') return NULL;

    for (int i = 0; i < vp->n_presets; i++) {
        if (strcasecmp(vp->names + i * TTS_PRESET_NAME_LEN, name) == 0) {
            return vp->embeds + i * vp->embed_dim;
        }
    }
    return NULL;
}
