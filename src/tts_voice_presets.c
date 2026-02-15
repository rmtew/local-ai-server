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

    /* Read count */
    int32_t n = 0;
    if (fread(&n, sizeof(int32_t), 1, f) != 1 || n <= 0 || n > 10000) {
        fclose(f);
        return -1;
    }

    /* Allocate and read presets */
    vp->presets = (tts_voice_preset_t *)calloc((size_t)n, sizeof(tts_voice_preset_t));
    if (!vp->presets) {
        fclose(f);
        return -1;
    }

    size_t expected = (size_t)n * sizeof(tts_voice_preset_t);
    if (fread(vp->presets, 1, expected, f) != expected) {
        free(vp->presets);
        vp->presets = NULL;
        fclose(f);
        return -1;
    }

    fclose(f);
    vp->n_presets = n;
    return 0;
}

void tts_voice_presets_free(tts_voice_presets_t *vp) {
    free(vp->presets);
    memset(vp, 0, sizeof(*vp));
}

const float *tts_voice_presets_find(const tts_voice_presets_t *vp, const char *name) {
    if (!vp->presets || !name || name[0] == '\0') return NULL;

    for (int i = 0; i < vp->n_presets; i++) {
        if (strcasecmp(vp->presets[i].name, name) == 0) {
            return vp->presets[i].embed;
        }
    }
    return NULL;
}
