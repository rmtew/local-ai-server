/*
 * ASR Request Handler
 * Routes HTTP requests to appropriate handlers.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "handler_asr.h"
#include "handler_tts.h"
#include "multipart.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* qwen-asr headers */
#include "qwen_asr.h"
#include "qwen_asr_audio.h"

/* ---- Route: GET /health ---- */

static void handle_health(SOCKET client) {
    const char *body = "{\"status\":\"ok\"}";
    http_send_response(client, 200, "application/json", body, strlen(body));
}

/* ---- Route: GET /v1/models ---- */

static void handle_models(SOCKET client, HandlerContext *ctx) {
    char buf[1024];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));
    jw_object_start(&w);
    jw_field_array_start(&w, "data");
    if (ctx->asr_ctx) {
        jw_array_sep(&w);
        jw_object_start(&w);
        jw_field_string(&w, "id", "qwen-asr");
        jw_field_string(&w, "object", "model");
        jw_object_end(&w);
    }
#ifdef USE_ORT
    if (ctx->tts) {
        jw_array_sep(&w);
        jw_object_start(&w);
        jw_field_string(&w, "id", "qwen3-tts");
        jw_field_string(&w, "object", "model");
        jw_object_end(&w);
    }
#endif
    jw_array_end(&w);
    jw_object_end(&w);

    http_send_response(client, 200, "application/json", buf, jw_length(&w));
}

/* ---- SSE streaming support ---- */

typedef struct {
    SOCKET client;
    qwen_ctx_t *asr_ctx;
    int error;
} SseTokenCtx;

/* Token callback: fires during qwen_transcribe_audio for each decoded token.
 * Reads the most recent timestamp entry (recorded before callback fires). */
static void sse_token_callback(const char *piece, void *userdata) {
    SseTokenCtx *sctx = (SseTokenCtx *)userdata;
    if (sctx->error) return;

    int audio_ms = 0;
    int byte_offset = 0;
    if (sctx->asr_ctx->token_ts_len > 0) {
        int idx = sctx->asr_ctx->token_ts_len - 1;
        audio_ms = sctx->asr_ctx->token_audio_ms[idx];
        byte_offset = sctx->asr_ctx->token_byte_offsets[idx];
    }

    char buf[1024];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));
    jw_object_start(&w);
    jw_field_string(&w, "token", piece);
    jw_field_int(&w, "audio_ms", audio_ms);
    jw_field_int(&w, "byte_offset", byte_offset);
    jw_object_end(&w);

    /* Send SSE event; mark error on failure so we skip further writes */
    int n = send(sctx->client, "data: ", 6, 0);
    if (n <= 0) { sctx->error = 1; return; }
    n = send(sctx->client, buf, (int)jw_length(&w), 0);
    if (n <= 0) { sctx->error = 1; return; }
    n = send(sctx->client, "\n\n", 2, 0);
    if (n <= 0) { sctx->error = 1; return; }
}

/* ---- Route: POST /v1/audio/transcriptions ---- */

static void handle_transcription(SOCKET client, const HttpRequest *request,
                                 HandlerContext *ctx) {
    /* Check ASR is loaded */
    if (!ctx->asr_ctx) {
        http_send_json_error(client, 501,
            "ASR not loaded (start server with --model=<dir>)",
            "not_implemented");
        return;
    }

    /* Require multipart/form-data */
    if (strstr(request->content_type, "multipart/form-data") == NULL) {
        http_send_json_error(client, 400,
            "Content-Type must be multipart/form-data",
            "invalid_request_error");
        return;
    }

    /* Extract boundary */
    char boundary[256];
    size_t bnd_len = multipart_get_boundary(request->content_type, boundary, sizeof(boundary));
    if (bnd_len == 0) {
        http_send_json_error(client, 400,
            "Missing multipart boundary",
            "invalid_request_error");
        return;
    }

    /* Parse multipart body */
    MultipartPart parts[MULTIPART_MAX_PARTS];
    int nparts = multipart_parse(boundary, bnd_len,
                                 request->body, request->body_len,
                                 parts, MULTIPART_MAX_PARTS);
    if (nparts <= 0) {
        http_send_json_error(client, 400,
            "Failed to parse multipart body",
            "invalid_request_error");
        return;
    }

    /* Find required "file" part */
    const MultipartPart *file_part = multipart_find(parts, nparts, "file");
    if (!file_part || file_part->data_len == 0) {
        http_send_json_error(client, 400,
            "Missing required 'file' field",
            "invalid_request_error");
        return;
    }

    /* Optional fields */
    char language[64] = {0};
    char response_format[32] = {0};
    char prompt[4096] = {0};

    const MultipartPart *lang_part = multipart_find(parts, nparts, "language");
    if (lang_part && lang_part->data_len > 0 && lang_part->data_len < sizeof(language)) {
        memcpy(language, lang_part->data, lang_part->data_len);
        language[lang_part->data_len] = '\0';
    }

    const MultipartPart *fmt_part = multipart_find(parts, nparts, "response_format");
    if (fmt_part && fmt_part->data_len > 0 && fmt_part->data_len < sizeof(response_format)) {
        memcpy(response_format, fmt_part->data, fmt_part->data_len);
        response_format[fmt_part->data_len] = '\0';
    }

    const MultipartPart *prompt_part = multipart_find(parts, nparts, "prompt");
    if (prompt_part && prompt_part->data_len > 0 && prompt_part->data_len < sizeof(prompt)) {
        memcpy(prompt, prompt_part->data, prompt_part->data_len);
        prompt[prompt_part->data_len] = '\0';
    }

    int verbose_json = (strcmp(response_format, "verbose_json") == 0);
    int streaming = (strcmp(response_format, "streaming_verbose_json") == 0);

    if (ctx->verbose) {
        printf("  file: %s (%zu bytes), language: %s, format: %s, prompt: %s\n",
               file_part->filename[0] ? file_part->filename : "(unnamed)",
               file_part->data_len,
               language[0] ? language : "(auto)",
               response_format[0] ? response_format : "json",
               prompt[0] ? prompt : "(none)");
    }

    /* Set language if specified (restore after) */
    char *prev_language = NULL;
    if (language[0]) {
        /* Save previous language setting */
        if (ctx->asr_ctx->force_language) {
            prev_language = _strdup(ctx->asr_ctx->force_language);
        }
        if (qwen_set_force_language(ctx->asr_ctx, language) != 0) {
            http_send_json_error(client, 400,
                "Unsupported language",
                "invalid_request_error");
            free(prev_language);
            return;
        }
    }

    /* Decode WAV from buffer */
    int n_samples = 0;
    float *samples = qwen_parse_wav_buffer(file_part->data, file_part->data_len, &n_samples);
    if (!samples || n_samples <= 0) {
        http_send_json_error(client, 400,
            "Failed to decode audio file (WAV format required)",
            "invalid_request_error");
        /* Restore language */
        if (language[0]) {
            qwen_set_force_language(ctx->asr_ctx, prev_language);
            free(prev_language);
        }
        return;
    }

    if (ctx->verbose) {
        printf("  audio: %.1f seconds (%d samples)\n",
               (double)n_samples / 16000.0, n_samples);
    }

    /* Set prompt bias if specified */
    if (prompt[0]) {
        qwen_set_prompt(ctx->asr_ctx, prompt);
    }

    /* --- Streaming SSE path --- */
    if (streaming) {
        http_send_sse_headers(client);

        SseTokenCtx sctx = { client, ctx->asr_ctx, 0 };
        qwen_set_token_callback(ctx->asr_ctx, sse_token_callback, &sctx);

        char *text = qwen_transcribe_audio(ctx->asr_ctx, samples, n_samples);
        free(samples);

        qwen_set_token_callback(ctx->asr_ctx, NULL, NULL);

        /* Restore prompt and language */
        if (prompt[0]) qwen_set_prompt(ctx->asr_ctx, NULL);
        if (language[0]) {
            qwen_set_force_language(ctx->asr_ctx, prev_language);
            free(prev_language);
        }

        /* Send done event with full verbose_json payload */
        if (text && !sctx.error) {
            const qwen_token_ts_t *ts = NULL;
            int ts_count = 0;
            qwen_get_token_timestamps(ctx->asr_ctx, &ts, &ts_count);

            size_t buf_size = 4096 + (size_t)ts_count * 256;
            char *json_buf = (char *)malloc(buf_size);
            if (json_buf) {
                JsonWriter w;
                jw_init(&w, json_buf, buf_size);
                jw_object_start(&w);
                jw_field_bool(&w, "done", 1);
                jw_field_string(&w, "text", text);
                jw_field_double(&w, "duration",
                                ctx->asr_ctx->perf_audio_ms / 1000.0, 3);
                jw_field_int(&w, "perf_total_ms",
                             (int64_t)ctx->asr_ctx->perf_total_ms);
                jw_field_int(&w, "perf_encode_ms",
                             (int64_t)ctx->asr_ctx->perf_encode_ms);
                jw_field_int(&w, "perf_decode_ms",
                             (int64_t)ctx->asr_ctx->perf_decode_ms);

                if (ts && ts_count > 0) {
                    jw_field_array_start(&w, "words");
                    for (int i = 0; i < ts_count; i++) {
                        jw_array_sep(&w);
                        jw_object_start(&w);

                        int start_off = ts[i].byte_offset;
                        int end_off = (i + 1 < ts_count)
                            ? ts[i + 1].byte_offset : (int)strlen(text);
                        int word_len = end_off - start_off;

                        const char *word_start = text + start_off;
                        while (word_len > 0 && (*word_start == ' '
                               || *word_start == '\t')) {
                            word_start++;
                            word_len--;
                        }
                        while (word_len > 0 && (word_start[word_len - 1] == ' '
                               || word_start[word_len - 1] == '\t')) {
                            word_len--;
                        }

                        char word_buf[512];
                        if (word_len >= (int)sizeof(word_buf))
                            word_len = (int)sizeof(word_buf) - 1;
                        if (word_len > 0)
                            memcpy(word_buf, word_start, (size_t)word_len);
                        word_buf[word_len > 0 ? word_len : 0] = '\0';

                        jw_field_string(&w, "word", word_buf);
                        jw_field_double(&w, "start",
                                        ts[i].audio_ms / 1000.0, 3);
                        double end_time = (i + 1 < ts_count)
                            ? ts[i + 1].audio_ms / 1000.0
                            : ctx->asr_ctx->perf_audio_ms / 1000.0;
                        jw_field_double(&w, "end", end_time, 3);
                        jw_field_int(&w, "byte_offset", ts[i].byte_offset);
                        jw_field_int(&w, "audio_ms", ts[i].audio_ms);

                        jw_object_end(&w);
                    }
                    jw_array_end(&w);
                }

                jw_object_end(&w);
                http_send_sse_event(client, json_buf, jw_length(&w));
                free(json_buf);
            }
        }

        if (ctx->verbose && text) {
            printf("  result: %s\n", text);
            printf("  perf: %.0f ms total, %.0f ms encode, %.0f ms decode\n",
                   ctx->asr_ctx->perf_total_ms,
                   ctx->asr_ctx->perf_encode_ms,
                   ctx->asr_ctx->perf_decode_ms);
        }

        free(text);
        return;
    }

    /* --- Non-streaming path --- */
    char *text = qwen_transcribe_audio(ctx->asr_ctx, samples, n_samples);
    free(samples);

    /* Restore prompt and language */
    if (prompt[0]) {
        qwen_set_prompt(ctx->asr_ctx, NULL);
    }
    if (language[0]) {
        qwen_set_force_language(ctx->asr_ctx, prev_language);
        free(prev_language);
    }

    if (!text) {
        http_send_json_error(client, 500,
            "Transcription failed",
            "server_error");
        return;
    }

    if (ctx->verbose) {
        printf("  result: %s\n", text);
        printf("  perf: %.0f ms total, %.0f ms encode, %.0f ms decode\n",
               ctx->asr_ctx->perf_total_ms,
               ctx->asr_ctx->perf_encode_ms,
               ctx->asr_ctx->perf_decode_ms);
    }

    /* Build JSON response */
    if (verbose_json) {
        /* Verbose format with word timestamps */
        const qwen_token_ts_t *ts = NULL;
        int ts_count = 0;
        qwen_get_token_timestamps(ctx->asr_ctx, &ts, &ts_count);

        /* Estimate buffer size: base + per-word entries */
        size_t buf_size = 4096 + (size_t)ts_count * 256;
        char *json_buf = (char *)malloc(buf_size);
        if (!json_buf) {
            free(text);
            http_send_json_error(client, 500, "Out of memory", "server_error");
            return;
        }

        JsonWriter w;
        jw_init(&w, json_buf, buf_size);
        jw_object_start(&w);
        jw_field_string(&w, "text", text);
        jw_field_double(&w, "duration", ctx->asr_ctx->perf_audio_ms / 1000.0, 3);
        jw_field_int(&w, "perf_total_ms", (int64_t)ctx->asr_ctx->perf_total_ms);
        jw_field_int(&w, "perf_encode_ms", (int64_t)ctx->asr_ctx->perf_encode_ms);
        jw_field_int(&w, "perf_decode_ms", (int64_t)ctx->asr_ctx->perf_decode_ms);

        if (ts && ts_count > 0) {
            jw_field_array_start(&w, "words");
            for (int i = 0; i < ts_count; i++) {
                jw_array_sep(&w);
                jw_object_start(&w);

                /* Extract word substring using byte offsets */
                int start_off = ts[i].byte_offset;
                int end_off = (i + 1 < ts_count) ? ts[i + 1].byte_offset : (int)strlen(text);
                int word_len = end_off - start_off;

                /* Trim leading/trailing whitespace from word */
                const char *word_start = text + start_off;
                while (word_len > 0 && (*word_start == ' ' || *word_start == '\t')) {
                    word_start++;
                    word_len--;
                }
                while (word_len > 0 && (word_start[word_len - 1] == ' ' ||
                       word_start[word_len - 1] == '\t')) {
                    word_len--;
                }

                /* Write word as null-terminated string */
                char word_buf[512];
                if (word_len >= (int)sizeof(word_buf)) word_len = (int)sizeof(word_buf) - 1;
                if (word_len > 0) {
                    memcpy(word_buf, word_start, (size_t)word_len);
                }
                word_buf[word_len > 0 ? word_len : 0] = '\0';

                jw_field_string(&w, "word", word_buf);
                jw_field_double(&w, "start", ts[i].audio_ms / 1000.0, 3);

                /* End time: next token's start, or audio duration for last token */
                double end_time = (i + 1 < ts_count)
                    ? ts[i + 1].audio_ms / 1000.0
                    : ctx->asr_ctx->perf_audio_ms / 1000.0;
                jw_field_double(&w, "end", end_time, 3);
                jw_field_int(&w, "byte_offset", ts[i].byte_offset);
                jw_field_int(&w, "audio_ms", ts[i].audio_ms);

                jw_object_end(&w);
            }
            jw_array_end(&w);
        }

        jw_object_end(&w);

        http_send_response(client, 200, "application/json", json_buf, jw_length(&w));
        free(json_buf);
    } else {
        /* Simple format: {"text":"..."} */
        size_t text_len = strlen(text);
        size_t buf_size = text_len * 2 + 64; /* room for escaping */
        char *json_buf = (char *)malloc(buf_size);
        if (!json_buf) {
            free(text);
            http_send_json_error(client, 500, "Out of memory", "server_error");
            return;
        }

        JsonWriter w;
        jw_init(&w, json_buf, buf_size);
        jw_object_start(&w);
        jw_field_string(&w, "text", text);
        jw_object_end(&w);

        http_send_response(client, 200, "application/json", json_buf, jw_length(&w));
        free(json_buf);
    }

    free(text);
}

/* ---- Main request router ---- */

void asr_handle_request(SOCKET client, const HttpRequest *request, void *user_data) {
    HandlerContext *ctx = (HandlerContext *)user_data;

    if (ctx->verbose) {
        printf("[%s] %s\n", request->method, request->path);
    }

    /* GET /health */
    if (strcmp(request->method, "GET") == 0 && strcmp(request->path, "/health") == 0) {
        handle_health(client);
        return;
    }

    /* GET /v1/models */
    if (strcmp(request->method, "GET") == 0 && strcmp(request->path, "/v1/models") == 0) {
        handle_models(client, ctx);
        return;
    }

    /* POST /v1/audio/transcriptions */
    if (strcmp(request->method, "POST") == 0 &&
        strcmp(request->path, "/v1/audio/transcriptions") == 0) {
        handle_transcription(client, request, ctx);
        return;
    }

    /* POST /v1/audio/speech */
    if (strcmp(request->method, "POST") == 0 &&
        strcmp(request->path, "/v1/audio/speech") == 0) {
        handle_tts_speech(client, request, ctx);
        return;
    }

    /* Handle OPTIONS for CORS preflight */
    if (strcmp(request->method, "OPTIONS") == 0) {
        http_send_response(client, 200, "text/plain", "", 0);
        return;
    }

    /* 404 for everything else */
    http_send_json_error(client, 404, "Not found", "invalid_request_error");
}
