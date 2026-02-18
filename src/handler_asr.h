/*
 * ASR Request Handler
 * Routes HTTP requests and handles transcription logic.
 */

#ifndef LOCAL_AI_HANDLER_ASR_H
#define LOCAL_AI_HANDLER_ASR_H

#include "http.h"
#include "qwen_asr.h"
#include "tts_pipeline.h"

/* Handler context passed as user_data to the HTTP handler callback */
typedef struct HandlerContext {
    qwen_ctx_t *asr_ctx;   /* NULL if ASR not loaded */
    int verbose;
    int threads;           /* Thread count for inference */
    TtsPipeline *tts;      /* NULL if TTS not loaded */

    /* Live streaming ASR session (one at a time) */
    struct {
        volatile int active;
        qwen_live_audio_t *live;
        SOCKET sse_socket;
#ifdef _WIN32
        HANDLE thread;
#else
        pthread_t thread;
#endif
        volatile int sse_error;
    } live_session;
} HandlerContext;

/* HTTP request handler (matches http_handler_fn signature) */
void asr_handle_request(SOCKET client, HttpRequest *request, void *user_data);

#endif /* LOCAL_AI_HANDLER_ASR_H */
