/*
 * TTS Request Handler
 * Parses OpenAI-compatible speech synthesis requests and returns WAV audio.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "handler_tts.h"
#include "handler_asr.h"
#include "json_reader.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void handle_tts_speech(SOCKET client, const HttpRequest *request,
                       struct HandlerContext *ctx) {
#ifndef USE_ORT
    (void)request;
    (void)ctx;
    http_send_json_error(client, 501,
        "TTS not available (server built without ONNX Runtime support)",
        "not_implemented");
#else
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
    double speed = 1.0;

    int input_len = jr_get_string((const char *)request->body, request->body_len,
                                   "input", input_text, sizeof(input_text));
    jr_get_string((const char *)request->body, request->body_len,
                   "voice", voice, sizeof(voice));
    jr_get_string((const char *)request->body, request->body_len,
                   "response_format", response_format, sizeof(response_format));
    jr_get_double((const char *)request->body, request->body_len,
                   "speed", &speed);

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

    if (ctx->verbose) {
        printf("  TTS request: input=\"%.60s%s\", voice=%s, speed=%.1f\n",
               input_text,
               strlen(input_text) > 60 ? "..." : "",
               voice[0] ? voice : "default",
               speed);
    }

    /* Run synthesis */
    TtsResult result;
    int rc = tts_pipeline_synthesize(ctx->tts, input_text,
                                      0.3f,    /* temperature */
                                      50,      /* top_k */
                                      (float)speed,
                                      &result);
    if (rc != 0) {
        http_send_json_error(client, 500,
            "Speech synthesis failed",
            "server_error");
        return;
    }

    if (ctx->verbose) {
        printf("  TTS result: %d steps, %d samples, %.0f ms, %.1f KB\n",
               result.n_steps, result.n_samples, result.elapsed_ms,
               (double)result.wav_len / 1024.0);
    }

    /* Send WAV response */
    http_send_response(client, 200, "audio/wav",
                       (const char *)result.wav_data, result.wav_len);
    free(result.wav_data);
#endif /* USE_ORT */
}
