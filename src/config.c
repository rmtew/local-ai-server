/*
 * config.c - Load defaults from config.json
 */

#define _CRT_SECURE_NO_WARNINGS
#include "config.h"
#include "json_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Try to open and read a file. Returns malloc'd content or NULL. */
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0 || sz > 64 * 1024) {  /* sanity: max 64KB config */
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    *out_len = n;
    return buf;
}

/* Try loading config from a specific path. Returns 0 on success. */
static int try_load(app_config_t *cfg, const char *path) {
    size_t len = 0;
    char *json = read_file(path, &len);
    if (!json) return -1;

    /* Extract fields (all optional) */
    jr_get_string(json, len, "model", cfg->model, sizeof(cfg->model));
    jr_get_string(json, len, "tts_model", cfg->tts_model, sizeof(cfg->tts_model));
    jr_get_string(json, len, "language", cfg->language, sizeof(cfg->language));

    double dval;
    if (jr_get_double(json, len, "port", &dval) == 0)
        cfg->port = (int)dval;
    if (jr_get_double(json, len, "threads", &dval) == 0)
        cfg->threads = (int)dval;
    if (jr_get_double(json, len, "tts_max_steps", &dval) == 0)
        cfg->tts_max_steps = (int)dval;

    int bval;
    if (jr_get_bool(json, len, "fp16", &bval) == 0)
        cfg->fp16 = bval;
    if (jr_get_bool(json, len, "verbose", &bval) == 0)
        cfg->verbose = bval;

    free(json);

    snprintf(cfg->config_path, sizeof(cfg->config_path), "%s", path);
    cfg->loaded = 1;
    return 0;
}

int config_load(app_config_t *cfg, const char *exe_path) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->fp16 = -1;
    cfg->verbose = -1;

    /* Try 1: cwd/config.json */
    if (try_load(cfg, "config.json") == 0)
        return 0;

    /* Try 2: exe's parent directory (bin/../config.json) */
    if (exe_path && exe_path[0]) {
        char dir[512];
        snprintf(dir, sizeof(dir), "%s", exe_path);

        /* Find last separator */
        char *sep = NULL;
        for (char *p = dir; *p; p++) {
            if (*p == '/' || *p == '\\') sep = p;
        }
        if (sep) {
            *sep = '\0';
            /* Go up one more level (from bin/ to project root) */
            sep = NULL;
            for (char *p = dir; *p; p++) {
                if (*p == '/' || *p == '\\') sep = p;
            }
            if (sep) {
                *sep = '\0';
                char path[512];
                snprintf(path, sizeof(path), "%s/config.json", dir);
                if (try_load(cfg, path) == 0)
                    return 0;
            }
        }
    }

    return -1;
}
