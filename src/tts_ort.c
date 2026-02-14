/*
 * tts_ort.c - ONNX Runtime wrapper for Qwen3-TTS models
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_ort.h"

#ifdef USE_ORT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

/* Check ORT status, print error and return -1 on failure */
#define ORT_CHECK(api, status) do { \
    if (status) { \
        const char *msg = (api)->GetErrorMessage(status); \
        fprintf(stderr, "ORT error: %s\n", msg); \
        (api)->ReleaseStatus(status); \
        return -1; \
    } \
} while (0)

/* Load a single ONNX session from model_dir/filename.
 * Returns 0 on success, -1 on failure. session may be NULL on skip. */
static int load_session(TtsOrt *ort, const char *model_dir,
                        const char *filename, OrtSession **out, int required) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", model_dir, filename);

    /* Check file exists */
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (required) {
            fprintf(stderr, "TTS: required model not found: %s\n", path);
            return -1;
        }
        if (ort->verbose) {
            printf("  TTS: optional model not found: %s (skipped)\n", path);
        }
        *out = NULL;
        return 0;
    }
    fclose(f);

    if (ort->verbose) {
        printf("  TTS: loading %s...\n", filename);
    }

    /* Convert to wide string for ORT API (Windows) */
    wchar_t wpath[512];
    size_t len = strlen(path);
    for (size_t i = 0; i <= len; i++) {
        wpath[i] = (wchar_t)(path[i] == '/' ? '\\' : path[i]);
    }

    OrtStatus *status = ort->api->CreateSession(ort->env, wpath, ort->opts, out);
    ORT_CHECK(ort->api, status);

    return 0;
}

/* Print session I/O info for debugging */
static void print_session_info(TtsOrt *ort, const char *name, OrtSession *session) {
    if (!session || !ort->verbose) return;

    OrtAllocator *allocator;
    ort->api->GetAllocatorWithDefaultOptions(&allocator);

    size_t n_inputs, n_outputs;
    ort->api->SessionGetInputCount(session, &n_inputs);
    ort->api->SessionGetOutputCount(session, &n_outputs);

    printf("  %s: %zu inputs, %zu outputs\n", name, n_inputs, n_outputs);

    for (size_t i = 0; i < n_inputs; i++) {
        char *input_name;
        ort->api->SessionGetInputName(session, i, allocator, &input_name);
        printf("    input[%zu]: %s\n", i, input_name);
        ort->api->AllocatorFree(allocator, input_name);
    }
    for (size_t i = 0; i < n_outputs; i++) {
        char *output_name;
        ort->api->SessionGetOutputName(session, i, allocator, &output_name);
        printf("    output[%zu]: %s\n", i, output_name);
        ort->api->AllocatorFree(allocator, output_name);
    }
}

int tts_ort_init(TtsOrt *ort, const char *model_dir, int verbose) {
    memset(ort, 0, sizeof(*ort));
    ort->verbose = verbose;

    /* Get the ORT API */
    const OrtApiBase *api_base = OrtGetApiBase();
    ort->api = api_base->GetApi(ORT_API_VERSION);
    if (!ort->api) {
        fprintf(stderr, "TTS: failed to get ORT API v%d\n", ORT_API_VERSION);
        return -1;
    }

    /* Create environment */
    OrtStatus *status;
    /* Use WARNING level to suppress noisy GetCpuPreferredNodes INFO spam */
    status = ort->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "tts", &ort->env);
    ORT_CHECK(ort->api, status);

    /* Create session options */
    status = ort->api->CreateSessionOptions(&ort->opts);
    ORT_CHECK(ort->api, status);

    ort->api->SetSessionGraphOptimizationLevel(ort->opts, ORT_ENABLE_ALL);
    ort->api->SetIntraOpNumThreads(ort->opts, 0); /* 0 = use all cores */

    /* Create CPU memory info */
    status = ort->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort->mem_info);
    ORT_CHECK(ort->api, status);

    printf("Loading TTS models from: %s\n", model_dir);

    /* Load required sessions */
    if (load_session(ort, model_dir, "text_project.onnx", &ort->text_project, 1) != 0) return -1;
    if (load_session(ort, model_dir, "codec_embed.onnx", &ort->codec_embed, 1) != 0) return -1;
    if (load_session(ort, model_dir, "talker_prefill.onnx", &ort->talker_prefill, 1) != 0) return -1;
    if (load_session(ort, model_dir, "talker_decode.onnx", &ort->talker_decode, 1) != 0) return -1;
    if (load_session(ort, model_dir, "code_predictor.onnx", &ort->code_predictor, 1) != 0) return -1;
    if (load_session(ort, model_dir, "code_predictor_embed.onnx", &ort->code_predictor_embed, 1) != 0) return -1;
    if (load_session(ort, model_dir, "tokenizer12hz_decode.onnx", &ort->tokenizer12hz_decode, 1) != 0) return -1;

    /* Phase 2 sessions (optional) */
    if (load_session(ort, model_dir, "speaker_encoder.onnx", &ort->speaker_encoder, 0) != 0) return -1;
    if (load_session(ort, model_dir, "tokenizer12hz_encode.onnx", &ort->tokenizer12hz_encode, 0) != 0) return -1;

    if (verbose) {
        print_session_info(ort, "text_project", ort->text_project);
        print_session_info(ort, "codec_embed", ort->codec_embed);
        print_session_info(ort, "talker_prefill", ort->talker_prefill);
        print_session_info(ort, "talker_decode", ort->talker_decode);
        print_session_info(ort, "code_predictor", ort->code_predictor);
        print_session_info(ort, "code_predictor_embed", ort->code_predictor_embed);
        print_session_info(ort, "tokenizer12hz_decode", ort->tokenizer12hz_decode);
    }

    /* Cache I/O names from sessions (queried once, used for all runs) */
    OrtAllocator *alloc;
    ort->api->GetAllocatorWithDefaultOptions(&alloc);

    /* Helper macro: query name from session, strdup into field */
    #define CACHE_NAME(session, idx, is_input, field) do { \
        char *_name; \
        if (is_input) \
            ort->api->SessionGetInputName(session, idx, alloc, &_name); \
        else \
            ort->api->SessionGetOutputName(session, idx, alloc, &_name); \
        ort->field = _strdup(_name); \
        ort->api->AllocatorFree(alloc, _name); \
    } while (0)

    /* text_project: 1 input, 1 output */
    CACHE_NAME(ort->text_project, 0, 1, text_project_in);
    CACHE_NAME(ort->text_project, 0, 0, text_project_out);

    /* codec_embed: 1 input, 1 output */
    CACHE_NAME(ort->codec_embed, 0, 1, codec_embed_in);
    CACHE_NAME(ort->codec_embed, 0, 0, codec_embed_out);

    /* code_predictor: 2 inputs, 1 output */
    CACHE_NAME(ort->code_predictor, 0, 1, code_pred_in_embeds);
    CACHE_NAME(ort->code_predictor, 1, 1, code_pred_in_step);
    CACHE_NAME(ort->code_predictor, 0, 0, code_pred_out);

    /* code_predictor_embed: 2 inputs, 1 output */
    CACHE_NAME(ort->code_predictor_embed, 0, 1, code_pred_embed_in_ids);
    CACHE_NAME(ort->code_predictor_embed, 1, 1, code_pred_embed_in_step);
    CACHE_NAME(ort->code_predictor_embed, 0, 0, code_pred_embed_out);

    /* tokenizer12hz_decode: 1 input, 1-2 outputs */
    CACHE_NAME(ort->tokenizer12hz_decode, 0, 1, vocoder_in);
    CACHE_NAME(ort->tokenizer12hz_decode, 0, 0, vocoder_out_audio);
    {
        size_t n_voc_out;
        ort->api->SessionGetOutputCount(ort->tokenizer12hz_decode, &n_voc_out);
        if (n_voc_out >= 2) {
            CACHE_NAME(ort->tokenizer12hz_decode, 1, 0, vocoder_out_lengths);
        }
    }

    #undef CACHE_NAME

    printf("TTS models loaded successfully.\n");
    return 0;
}

void tts_ort_free(TtsOrt *ort) {
    if (!ort->api) return;

    /* Free cached I/O name strings */
    free(ort->text_project_in); free(ort->text_project_out);
    free(ort->codec_embed_in); free(ort->codec_embed_out);
    free(ort->code_pred_in_embeds); free(ort->code_pred_in_step); free(ort->code_pred_out);
    free(ort->code_pred_embed_in_ids); free(ort->code_pred_embed_in_step); free(ort->code_pred_embed_out);
    free(ort->vocoder_in); free(ort->vocoder_out_audio); free(ort->vocoder_out_lengths);

    if (ort->text_project) ort->api->ReleaseSession(ort->text_project);
    if (ort->codec_embed) ort->api->ReleaseSession(ort->codec_embed);
    if (ort->talker_prefill) ort->api->ReleaseSession(ort->talker_prefill);
    if (ort->talker_decode) ort->api->ReleaseSession(ort->talker_decode);
    if (ort->code_predictor) ort->api->ReleaseSession(ort->code_predictor);
    if (ort->code_predictor_embed) ort->api->ReleaseSession(ort->code_predictor_embed);
    if (ort->speaker_encoder) ort->api->ReleaseSession(ort->speaker_encoder);
    if (ort->tokenizer12hz_encode) ort->api->ReleaseSession(ort->tokenizer12hz_encode);
    if (ort->tokenizer12hz_decode) ort->api->ReleaseSession(ort->tokenizer12hz_decode);

    if (ort->mem_info) ort->api->ReleaseMemoryInfo(ort->mem_info);
    if (ort->opts) ort->api->ReleaseSessionOptions(ort->opts);
    if (ort->env) ort->api->ReleaseEnv(ort->env);

    memset(ort, 0, sizeof(*ort));
}

OrtValue *tts_ort_tensor_int64(TtsOrt *ort, const int64_t *data, int64_t len) {
    int64_t shape[] = { len };
    OrtValue *tensor = NULL;
    OrtStatus *status = ort->api->CreateTensorWithDataAsOrtValue(
        ort->mem_info,
        (void *)data, (size_t)(len * sizeof(int64_t)),
        shape, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &tensor);
    if (status) {
        fprintf(stderr, "TTS: tensor_int64 error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return NULL;
    }
    return tensor;
}

OrtValue *tts_ort_tensor_int64_2d(TtsOrt *ort, const int64_t *data, int64_t len) {
    int64_t shape[] = { 1, len };
    OrtValue *tensor = NULL;
    OrtStatus *status = ort->api->CreateTensorWithDataAsOrtValue(
        ort->mem_info,
        (void *)data, (size_t)(len * sizeof(int64_t)),
        shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &tensor);
    if (status) {
        fprintf(stderr, "TTS: tensor_int64_2d error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return NULL;
    }
    return tensor;
}

OrtValue *tts_ort_tensor_float_3d(TtsOrt *ort, const float *data,
                                   int64_t d0, int64_t d1, int64_t d2) {
    int64_t shape[] = { d0, d1, d2 };
    size_t byte_len = (size_t)(d0 * d1 * d2) * sizeof(float);
    OrtValue *tensor = NULL;
    OrtStatus *status = ort->api->CreateTensorWithDataAsOrtValue(
        ort->mem_info,
        (void *)data, byte_len,
        shape, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &tensor);
    if (status) {
        fprintf(stderr, "TTS: tensor_float_3d error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return NULL;
    }
    return tensor;
}

OrtValue *tts_ort_tensor_float_2d(TtsOrt *ort, const float *data,
                                   int64_t d0, int64_t d1) {
    int64_t shape[] = { d0, d1 };
    size_t byte_len = (size_t)(d0 * d1) * sizeof(float);
    OrtValue *tensor = NULL;
    OrtStatus *status = ort->api->CreateTensorWithDataAsOrtValue(
        ort->mem_info,
        (void *)data, byte_len,
        shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &tensor);
    if (status) {
        fprintf(stderr, "TTS: tensor_float_2d error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return NULL;
    }
    return tensor;
}

int tts_ort_run(TtsOrt *ort, OrtSession *session,
                const char **input_names, const OrtValue *const *input_values, size_t n_inputs,
                const char **output_names, OrtValue **output_values, size_t n_outputs) {
    OrtStatus *status = ort->api->Run(
        session, NULL,
        input_names, input_values, n_inputs,
        output_names, n_outputs, output_values);
    if (status) {
        fprintf(stderr, "TTS: Run error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return -1;
    }
    return 0;
}

float *tts_ort_get_float_data(TtsOrt *ort, OrtValue *value) {
    float *data = NULL;
    OrtStatus *status = ort->api->GetTensorMutableData(value, (void **)&data);
    if (status) {
        fprintf(stderr, "TTS: GetTensorMutableData error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return NULL;
    }
    return data;
}

int64_t *tts_ort_get_int64_data(TtsOrt *ort, OrtValue *value) {
    int64_t *data = NULL;
    OrtStatus *status = ort->api->GetTensorMutableData(value, (void **)&data);
    if (status) {
        fprintf(stderr, "TTS: GetTensorMutableData error: %s\n", ort->api->GetErrorMessage(status));
        ort->api->ReleaseStatus(status);
        return NULL;
    }
    return data;
}

int tts_ort_get_shape(TtsOrt *ort, OrtValue *value, int64_t *shape, int max_dims) {
    OrtTensorTypeAndShapeInfo *info = NULL;
    OrtStatus *status = ort->api->GetTensorTypeAndShape(value, &info);
    if (status) {
        ort->api->ReleaseStatus(status);
        return 0;
    }

    size_t n_dims;
    ort->api->GetDimensionsCount(info, &n_dims);
    if ((int)n_dims > max_dims) n_dims = (size_t)max_dims;
    ort->api->GetDimensions(info, shape, n_dims);
    ort->api->ReleaseTensorTypeAndShapeInfo(info);
    return (int)n_dims;
}

void tts_ort_release_value(TtsOrt *ort, OrtValue *value) {
    if (value) ort->api->ReleaseValue(value);
}

#endif /* USE_ORT */
