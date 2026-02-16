/*
 * config.h - Load defaults from config.json
 *
 * Reads a flat JSON config file and provides defaults for model paths,
 * port, threads, etc. CLI arguments always override config values.
 *
 * Searches for config.json in: cwd, then exe's parent directory.
 */

#ifndef LOCAL_AI_CONFIG_H
#define LOCAL_AI_CONFIG_H

typedef struct {
    char model[512];        /* ASR model directory */
    char tts_model[512];    /* TTS model directory */
    int  port;              /* Listen port (0 = use default) */
    int  threads;           /* CPU threads (0 = use default) */
    int  tts_fp16;          /* -1 = not set, 0 = off, 1 = on (default: on for GPU builds) */
    int  asr_fp16;          /* -1 = not set, 0 = off, 1 = on (default: off) */
    int  verbose;           /* -1 = not set, 0 = off, 1 = on */
    int  tts_max_steps;     /* 0 = use default */
    char language[64];      /* ASR language override */
    int  loaded;            /* 1 if config was successfully loaded */
    char config_path[512];  /* Path that was loaded */
} app_config_t;

/* Load config.json, searching cwd then exe parent dir.
 * exe_path: argv[0] (used to find exe directory). Can be NULL.
 * Returns 0 on success (cfg populated), -1 if not found (cfg zeroed). */
int config_load(app_config_t *cfg, const char *exe_path);

#endif /* LOCAL_AI_CONFIG_H */
