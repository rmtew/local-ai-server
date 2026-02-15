/*
 * TTS Request Handler
 * Handles POST /v1/audio/speech requests.
 */

#ifndef LOCAL_AI_HANDLER_TTS_H
#define LOCAL_AI_HANDLER_TTS_H

#include "http.h"
#include "tts_pipeline.h"

/* Forward declaration -- full context defined in handler_asr.h */
struct HandlerContext;

/* Handle POST /v1/audio/speech.
 * Expects JSON body: {"input":"text","voice":"default","speed":1.0,"response_format":"wav"}
 * Returns audio/wav binary response. */
void handle_tts_speech(SOCKET client, const HttpRequest *request,
                       struct HandlerContext *ctx);

#endif /* LOCAL_AI_HANDLER_TTS_H */
