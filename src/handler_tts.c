/*
 * TTS Request Handler
 * Parses OpenAI-compatible speech synthesis requests and returns WAV audio.
 * Supports streaming via SSE when "stream":true is set in the request body.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "handler_tts.h"
#include "handler_asr.h"
#include "json_reader.h"
#include "json.h"
#include "qwen_asr_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

/* ---- Base64 encoder ---- */

static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* Encode binary data to base64. Returns number of chars written.
 * out must have room for at least 4*((len+2)/3) bytes. */
static size_t base64_encode(char *out, const unsigned char *in, size_t len) {
    size_t i, j = 0;
    for (i = 0; i + 2 < len; i += 3) {
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i+1] << 8) | in[i+2];
        out[j++] = b64_table[(v >> 18) & 0x3F];
        out[j++] = b64_table[(v >> 12) & 0x3F];
        out[j++] = b64_table[(v >> 6) & 0x3F];
        out[j++] = b64_table[v & 0x3F];
    }
    if (i < len) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i + 1 < len) v |= (uint32_t)in[i+1] << 8;
        out[j++] = b64_table[(v >> 18) & 0x3F];
        out[j++] = b64_table[(v >> 12) & 0x3F];
        out[j++] = (i + 1 < len) ? b64_table[(v >> 6) & 0x3F] : '=';
        out[j++] = '=';
    }
    return j;
}

/* ---- SSE streaming support ---- */

typedef struct {
    SOCKET client;
} tts_stream_ctx_t;

static void tts_stream_progress(const char *phase, int step, int max_steps,
                                void *userdata) {
    tts_stream_ctx_t *sctx = (tts_stream_ctx_t *)userdata;
    char buf[256];
    int len;
    if (strcmp(phase, "decoding") == 0) {
        len = snprintf(buf, sizeof(buf),
            "{\"phase\":\"decoding\",\"step\":%d,\"max_steps\":%d}",
            step, max_steps);
    } else {
        len = snprintf(buf, sizeof(buf), "{\"phase\":\"%s\"}", phase);
    }
    http_send_sse_event(sctx->client, buf, (size_t)len);
}

/* Send the final SSE event containing base64-encoded WAV audio and metadata. */
static void send_audio_sse(SOCKET client, const TtsResult *result) {
    size_t b64_len = 4 * ((result->wav_len + 2) / 3);
    /* {"phase":"done","n_steps":NNN,"n_samples":NNNNNNN,"elapsed_ms":NNNN.N,"audio":"..."}  */
    size_t prefix_max = 128;
    size_t total = prefix_max + b64_len + 4; /* 4 for closing "}\0 + margin */
    char *buf = (char *)malloc(total);
    if (!buf) return;

    int off = snprintf(buf, prefix_max,
        "{\"phase\":\"done\",\"n_steps\":%d,\"n_samples\":%d,"
        "\"elapsed_ms\":%.1f,\"audio\":\"",
        result->n_steps, result->n_samples, result->elapsed_ms);

    off += (int)base64_encode(buf + off, result->wav_data, result->wav_len);
    buf[off++] = '"';
    buf[off++] = '}';

    http_send_sse_event(client, buf, (size_t)off);
    free(buf);
}

/* ---- Main handler ---- */

void handle_tts_speech(SOCKET client, const HttpRequest *request,
                       struct HandlerContext *ctx) {
    if (!ctx->tts) {
        http_send_json_error(client, 501,
            "TTS not loaded (start server with --tts-model=<dir>)",
            "not_implemented");
        return;
    }

    /* Require JSON content type (allow empty for curl default) */
    if (request->content_type[0] != '\0' &&
        strstr(request->content_type, "application/json") == NULL) {
        http_send_json_error(client, 400,
            "Content-Type must be application/json",
            "invalid_request_error");
        return;
    }

    if (!request->body || request->body_len == 0) {
        http_send_json_error(client, 400,
            "Request body is empty",
            "invalid_request_error");
        return;
    }

    /* Parse JSON body */
    char input_text[4096] = {0};
    char voice[64] = {0};
    char response_format[32] = {0};
    char language[32] = {0};
    double speed = 1.0;
    double seed = -1.0;
    double temperature = 0.9;
    double top_k_d = 50.0;
    int stream = 0;

    int input_len = jr_get_string((const char *)request->body, request->body_len,
                                   "input", input_text, sizeof(input_text));
    jr_get_string((const char *)request->body, request->body_len,
                   "voice", voice, sizeof(voice));
    jr_get_string((const char *)request->body, request->body_len,
                   "response_format", response_format, sizeof(response_format));
    jr_get_string((const char *)request->body, request->body_len,
                   "language", language, sizeof(language));
    jr_get_double((const char *)request->body, request->body_len,
                   "speed", &speed);
    jr_get_double((const char *)request->body, request->body_len,
                   "seed", &seed);
    jr_get_double((const char *)request->body, request->body_len,
                   "temperature", &temperature);
    jr_get_double((const char *)request->body, request->body_len,
                   "top_k", &top_k_d);
    jr_get_bool((const char *)request->body, request->body_len,
                 "stream", &stream);

    if (input_len <= 0 || input_text[0] == '\0') {
        http_send_json_error(client, 400,
            "Missing required 'input' field",
            "invalid_request_error");
        return;
    }

    /* Validate response_format (only wav supported) */
    if (response_format[0] != '\0' && strcmp(response_format, "wav") != 0 &&
        strcmp(response_format, "pcm") != 0) {
        http_send_json_error(client, 400,
            "Unsupported response_format (only 'wav' is supported)",
            "invalid_request_error");
        return;
    }

    int top_k = (int)top_k_d;
    if (top_k < 1) top_k = 50;

    if (ctx->verbose) {
        printf("  TTS request: input=\"%.60s%s\", voice=%s, lang=%s, temp=%.2f, top_k=%d, speed=%.1f, seed=%d, stream=%d\n",
               input_text,
               strlen(input_text) > 60 ? "..." : "",
               voice[0] ? voice : "default",
               language[0] ? language : "auto",
               temperature,
               top_k,
               speed,
               seed >= 0.0 ? (int)seed : -1,
               stream);
    }

    /* Seed RNG for each request.  Without this, rand() state carries over
     * from previous synthesis calls, and certain states cause the model to
     * never emit EOS (running to max_steps = 16s of audio).
     * When an explicit seed is requested, also force single-threaded GEMM
     * for deterministic output. */
    int saved_threads = 0;
    if (seed >= 0.0) {
        srand((unsigned int)seed);
        saved_threads = ctx->threads;
        qwen_set_threads(1);
    } else {
        /* Time-based seed: mix high-resolution timer bits for good entropy */
        unsigned int t = (unsigned int)(platform_time_ms() * 1000.0);
        srand(t);
    }

    /* Set up streaming if requested */
    tts_stream_ctx_t sctx;
    tts_progress_fn progress_fn = NULL;
    void *progress_data = NULL;
    if (stream) {
        http_send_sse_headers(client);
        sctx.client = client;
        progress_fn = tts_stream_progress;
        progress_data = &sctx;
    }

    /* Run synthesis */
    double t_synth_start = platform_time_ms();
    TtsResult result;
    const char *v = voice[0] ? voice : NULL;
    const char *lang = language[0] ? language : NULL;
    int rc = tts_pipeline_synthesize(ctx->tts, input_text, v, lang,
                                      (float)temperature, top_k,
                                      (float)speed,
                                      progress_fn, progress_data,
                                      &result);
    double t_synth_end = platform_time_ms();

    if (saved_threads > 0) {
        qwen_set_threads(saved_threads);
    }

    if (rc != 0) {
        if (stream) {
            const char *err = "{\"error\":\"Speech synthesis failed\"}";
            http_send_sse_event(client, err, strlen(err));
            http_send_sse_event(client, "[DONE]", 6);
        } else {
            http_send_json_error(client, 500,
                "Speech synthesis failed",
                "server_error");
        }
        return;
    }

    if (stream) {
        /* Send audio as base64 in final SSE event, then [DONE] */
        send_audio_sse(client, &result);
        http_send_sse_event(client, "[DONE]", 6);
    } else {
        /* Send WAV response */
        http_send_response(client, 200, "audio/wav",
                           (const char *)result.wav_data, result.wav_len);
    }
    double t_send_end = platform_time_ms();

    printf("  TTS: %d steps, %d samples (%.1fs), synth=%.0fms send=%.0fms total=%.0fms (%.1f KB)\n",
           result.n_steps, result.n_samples,
           (double)result.n_samples / 24000.0,
           t_synth_end - t_synth_start,
           t_send_end - t_synth_end,
           t_send_end - t_synth_start,
           (double)result.wav_len / 1024.0);

    free(result.wav_data);
}
