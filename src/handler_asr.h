/*
 * ASR Request Handler
 * Routes HTTP requests and handles transcription logic.
 */

#ifndef LOCAL_AI_HANDLER_ASR_H
#define LOCAL_AI_HANDLER_ASR_H

#include "http.h"
#include "qwen_asr.h"

#ifdef USE_ORT
#include "tts_pipeline.h"
#endif

/* Handler context passed as user_data to the HTTP handler callback */
typedef struct HandlerContext {
    qwen_ctx_t *asr_ctx;   /* NULL if ASR not loaded */
    int verbose;
    int threads;           /* Thread count for inference */
#ifdef USE_ORT
    TtsPipeline *tts;      /* NULL if TTS not loaded */
#endif
} HandlerContext;

/* HTTP request handler (matches http_handler_fn signature) */
void asr_handle_request(SOCKET client, const HttpRequest *request, void *user_data);

#endif /* LOCAL_AI_HANDLER_ASR_H */
