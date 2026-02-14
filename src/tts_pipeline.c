/*
 * tts_pipeline.c - Qwen3-TTS synthesis pipeline
 *
 * Implements the full TTS pipeline:
 *   1. BPE tokenize input text
 *   2. Build embeddings (text_project + codec_embed)
 *   3. Talker prefill -> logits + KV cache
 *   4. Autoregressive decode loop with code predictor
 *   5. Vocoder decode (tokenizer12hz_decode) -> audio
 *   6. WAV encoding
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_pipeline.h"

#ifdef USE_ORT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "onnxruntime_c_api.h"

/* From tts_sampling.c */
extern void tts_apply_repetition_penalty(float *logits, int vocab_size,
                                         const int *history, int history_len,
                                         float penalty);
extern int tts_sample_topk(float *logits, int vocab_size,
                           float temperature, int top_k);

/* ---- Constants from Qwen3-TTS 0.6B config.json ---- */

#define TTS_BOS_TOKEN_ID    151672
#define TTS_EOS_TOKEN_ID    151673
#define TTS_PAD_TOKEN_ID    151671
#define IM_START_TOKEN_ID   151644
#define IM_END_TOKEN_ID     151645

#define CODEC_BOS_ID        2149
#define CODEC_EOS_TOKEN_ID  2150
#define CODEC_PAD_ID        2148
#define CODEC_NOTHINK_ID    2155
#define CODEC_THINK_BOS_ID  2156
#define CODEC_THINK_EOS_ID  2157

#define NUM_CODE_GROUPS     16
#define HIDDEN_SIZE         1024
#define TALKER_VOCAB_SIZE   3072
#define TALKER_NUM_LAYERS   28
#define NUM_KV_TENSORS      (TALKER_NUM_LAYERS * 2)  /* 56: key+value per layer */

#define MAX_DECODE_STEPS    2048
#define WAV_SAMPLE_RATE     24000

/* ---- WAV encoding ---- */

static void write_le16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static void write_le32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

/* Encode float audio samples as 16-bit PCM WAV file.
 * Returns malloc'd WAV data, sets *out_len. */
static unsigned char *encode_wav(const float *samples, int n_samples, size_t *out_len) {
    size_t data_size = (size_t)n_samples * 2;  /* 16-bit = 2 bytes per sample */
    size_t wav_size = 44 + data_size;
    unsigned char *wav = (unsigned char *)malloc(wav_size);
    if (!wav) return NULL;

    /* RIFF header */
    memcpy(wav, "RIFF", 4);
    write_le32(wav + 4, (uint32_t)(wav_size - 8));
    memcpy(wav + 8, "WAVE", 4);

    /* fmt chunk */
    memcpy(wav + 12, "fmt ", 4);
    write_le32(wav + 16, 16);                    /* chunk size */
    write_le16(wav + 20, 1);                     /* PCM format */
    write_le16(wav + 22, 1);                     /* mono */
    write_le32(wav + 24, WAV_SAMPLE_RATE);       /* sample rate */
    write_le32(wav + 28, WAV_SAMPLE_RATE * 2);   /* byte rate */
    write_le16(wav + 32, 2);                     /* block align */
    write_le16(wav + 34, 16);                    /* bits per sample */

    /* data chunk */
    memcpy(wav + 36, "data", 4);
    write_le32(wav + 40, (uint32_t)data_size);

    /* Convert float [-1,1] to int16 */
    int16_t *pcm = (int16_t *)(wav + 44);
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm[i] = (int16_t)(s * 32767.0f);
    }

    *out_len = wav_size;
    return wav;
}

/* ---- Pipeline initialization ---- */

int tts_pipeline_init(TtsPipeline *tts, const char *model_dir, int verbose) {
    memset(tts, 0, sizeof(*tts));
    tts->verbose = verbose;

    /* Load ONNX sessions */
    if (tts_ort_init(&tts->ort, model_dir, verbose) != 0) {
        return -1;
    }

    /* Load BPE tokenizer (same Qwen vocab as ASR) */
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    tts->tokenizer = qwen_tokenizer_load(vocab_path);
    if (!tts->tokenizer) {
        fprintf(stderr, "TTS: failed to load tokenizer from %s\n", vocab_path);
        tts_ort_free(&tts->ort);
        return -1;
    }

    if (verbose) {
        printf("TTS tokenizer loaded: %d tokens\n", tts->tokenizer->vocab_size);
    }

    return 0;
}

void tts_pipeline_free(TtsPipeline *tts) {
    if (tts->tokenizer) {
        qwen_tokenizer_free(tts->tokenizer);
        tts->tokenizer = NULL;
    }
    tts_ort_free(&tts->ort);
}

/* ---- Helper: run text_project on token IDs ---- */

/* Run text_project.onnx: int64 IDs [1, N] -> float embeddings [1, N, 1024] */
static OrtValue *run_text_project(TtsPipeline *tts, const int64_t *ids, int n_ids) {
    OrtValue *input = tts_ort_tensor_int64_2d(&tts->ort, ids, (int64_t)n_ids);
    if (!input) return NULL;

    const char *in_names[] = { tts->ort.text_project_in };
    const char *out_names[] = { tts->ort.text_project_out };
    OrtValue *output = NULL;

    int rc = tts_ort_run(&tts->ort, tts->ort.text_project,
                          in_names, (const OrtValue *const *)&input, 1,
                          out_names, &output, 1);
    tts_ort_release_value(&tts->ort, input);
    if (rc != 0) return NULL;
    return output;
}

/* Run codec_embed.onnx: int64 IDs [1, N] -> float embeddings [1, N, 1024] */
static OrtValue *run_codec_embed(TtsPipeline *tts, const int64_t *ids, int n_ids) {
    OrtValue *input = tts_ort_tensor_int64_2d(&tts->ort, ids, (int64_t)n_ids);
    if (!input) return NULL;

    const char *in_names[] = { tts->ort.codec_embed_in };
    const char *out_names[] = { tts->ort.codec_embed_out };
    OrtValue *output = NULL;

    int rc = tts_ort_run(&tts->ort, tts->ort.codec_embed,
                          in_names, (const OrtValue *const *)&input, 1,
                          out_names, &output, 1);
    tts_ort_release_value(&tts->ort, input);
    if (rc != 0) return NULL;
    return output;
}

/* ---- Build input token sequence ---- */

/* Build the input_ids for the assistant template.
 * Format: <|im_start|>assistant\n{text}<|im_end|> + padding
 *
 * Returns malloc'd int64 array. Sets *out_n_ids, *out_role_len (should be 3),
 * and *out_text_start (index of first text token).
 * The last 5 tokens are end padding (sliced off by the pipeline). */
static int64_t *build_input_ids(TtsPipeline *tts, const char *text,
                                int *out_n_ids, int *out_role_len) {
    /* Encode "assistant\n" to get role continuation tokens */
    int n_role_cont = 0;
    int *role_cont = qwen_tokenizer_encode(tts->tokenizer, "assistant\n", &n_role_cont);
    if (!role_cont || n_role_cont < 1) {
        fprintf(stderr, "TTS: failed to encode role tokens\n");
        free(role_cont);
        return NULL;
    }

    /* Encode the user text */
    int n_text = 0;
    int *text_tokens = qwen_tokenizer_encode(tts->tokenizer, text, &n_text);
    if (!text_tokens || n_text < 1) {
        fprintf(stderr, "TTS: failed to encode text\n");
        free(role_cont);
        free(text_tokens);
        return NULL;
    }

    /* Total: 1 (im_start) + role_cont + text + 5 (end padding) */
    int role_len = 1 + n_role_cont;  /* im_start + "assistant\n" */
    int total = role_len + n_text + 5;
    int64_t *ids = (int64_t *)malloc((size_t)total * sizeof(int64_t));
    if (!ids) {
        free(role_cont);
        free(text_tokens);
        return NULL;
    }

    int pos = 0;

    /* Role tokens */
    ids[pos++] = IM_START_TOKEN_ID;
    for (int i = 0; i < n_role_cont; i++) {
        ids[pos++] = (int64_t)role_cont[i];
    }

    /* Text tokens */
    for (int i = 0; i < n_text; i++) {
        ids[pos++] = (int64_t)text_tokens[i];
    }

    /* End padding (5 tokens, sliced off by pipeline) */
    ids[pos++] = TTS_EOS_TOKEN_ID;
    ids[pos++] = IM_END_TOKEN_ID;
    ids[pos++] = IM_END_TOKEN_ID;  /* padding */
    ids[pos++] = IM_END_TOKEN_ID;
    ids[pos++] = IM_END_TOKEN_ID;

    free(role_cont);
    free(text_tokens);

    *out_n_ids = total;
    *out_role_len = role_len;
    return ids;
}

/* ---- Build prefill embeddings ---- */

/* Extract float data from OrtValue, copy to buffer. Returns number of floats copied. */
static int extract_floats(TtsPipeline *tts, OrtValue *val, float *dst, int max_floats) {
    int64_t shape[4];
    int ndims = tts_ort_get_shape(&tts->ort, val, shape, 4);
    int64_t total = 1;
    for (int i = 0; i < ndims; i++) total *= shape[i];
    if (total > max_floats) total = max_floats;

    float *src = tts_ort_get_float_data(&tts->ort, val);
    if (!src) return 0;
    memcpy(dst, src, (size_t)total * sizeof(float));
    return (int)total;
}

/* Build the prefill input embeddings and trailing text hidden states.
 *
 * For no-voice-clone, auto-language:
 *   codec_prefill = [nothink, think_bos, think_eos, pad, bos]
 *   prefill_embed = [role_embed, (pad*3 + tts_bos) + codec_prefill[:4], text_first + codec_prefill[4]]
 *   trailing = [text_project(remaining_text), tts_eos_embed]
 *
 * Returns 0 on success. Caller must free *prefill_data and *trailing_data. */
static int build_prefill_embeddings(TtsPipeline *tts,
                                    const int64_t *input_ids, int n_ids, int role_len,
                                    float **prefill_data, int *prefill_seq_len,
                                    float **trailing_data, int *trailing_seq_len,
                                    float **tts_pad_embed_out) {
    int H = HIDDEN_SIZE;

    /* 1. Get TTS special token embeddings: text_project([tts_bos, tts_eos, tts_pad]) */
    int64_t tts_special_ids[] = { TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID };
    OrtValue *tts_special_val = run_text_project(tts, tts_special_ids, 3);
    if (!tts_special_val) return -1;

    float *tts_special = tts_ort_get_float_data(&tts->ort, tts_special_val);
    if (!tts_special) { tts_ort_release_value(&tts->ort, tts_special_val); return -1; }

    float *tts_bos_embed = tts_special;              /* [1024] */
    float *tts_eos_embed = tts_special + H;          /* [1024] */
    float *tts_pad_embed = tts_special + 2 * H;      /* [1024] */

    /* Save tts_pad_embed for decode loop */
    *tts_pad_embed_out = (float *)malloc((size_t)H * sizeof(float));
    if (!*tts_pad_embed_out) { tts_ort_release_value(&tts->ort, tts_special_val); return -1; }
    memcpy(*tts_pad_embed_out, tts_pad_embed, (size_t)H * sizeof(float));

    /* 2. Get role embeddings: text_project(input_ids[:role_len]) */
    OrtValue *role_val = run_text_project(tts, input_ids, role_len);
    if (!role_val) { tts_ort_release_value(&tts->ort, tts_special_val); return -1; }
    float *role_embed = tts_ort_get_float_data(&tts->ort, role_val);

    /* 3. Get codec prefix embeddings */
    int64_t codec_prefix_ids[] = { CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID,
                                    CODEC_PAD_ID, CODEC_BOS_ID };
    OrtValue *codec_val = run_codec_embed(tts, codec_prefix_ids, 5);
    if (!codec_val) {
        tts_ort_release_value(&tts->ort, tts_special_val);
        tts_ort_release_value(&tts->ort, role_val);
        return -1;
    }
    float *codec_embed = tts_ort_get_float_data(&tts->ort, codec_val);

    /* 4. Get first text token embedding: text_project(input_ids[role_len:role_len+1]) */
    OrtValue *text_first_val = run_text_project(tts, &input_ids[role_len], 1);
    if (!text_first_val) {
        tts_ort_release_value(&tts->ort, tts_special_val);
        tts_ort_release_value(&tts->ort, role_val);
        tts_ort_release_value(&tts->ort, codec_val);
        return -1;
    }
    float *text_first_embed = tts_ort_get_float_data(&tts->ort, text_first_val);

    /* 5. Build prefill embedding sequence:
     *    [role_embed(3), (pad*3 + tts_bos)(4) + codec[:4], text_first(1) + codec[4]]
     *    Total: role_len + 4 + 1 = role_len + 5 */
    int pf_len = role_len + 4 + 1;
    float *pf = (float *)calloc((size_t)pf_len * (size_t)H, sizeof(float));
    if (!pf) goto cleanup;

    /* Copy role embeddings */
    memcpy(pf, role_embed, (size_t)role_len * (size_t)H * sizeof(float));

    /* Build talker_embed = (pad*3 + tts_bos) + codec[:4] */
    int offset = role_len * H;
    /* pad_block (3 copies of tts_pad_embed) + tts_bos_embed */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < H; j++) {
            pf[offset + i * H + j] = tts_pad_embed[j] + codec_embed[i * H + j];
        }
    }
    /* tts_bos + codec[3] */
    for (int j = 0; j < H; j++) {
        pf[offset + 3 * H + j] = tts_bos_embed[j] + codec_embed[3 * H + j];
    }

    /* text_first + codec[4] (last codec embed) */
    offset = (role_len + 4) * H;
    for (int j = 0; j < H; j++) {
        pf[offset + j] = text_first_embed[j] + codec_embed[4 * H + j];
    }

    *prefill_data = pf;
    *prefill_seq_len = pf_len;

    /* 6. Build trailing text hidden states:
     *    text_project(input_ids[role_len+1 : n_ids-5]) + tts_eos_embed */
    int text_remaining_start = role_len + 1;
    int text_remaining_end = n_ids - 5;
    int n_remaining = text_remaining_end - text_remaining_start;

    if (n_remaining > 0) {
        OrtValue *trail_val = run_text_project(tts, &input_ids[text_remaining_start], n_remaining);
        if (!trail_val) goto cleanup;
        float *trail_embed = tts_ort_get_float_data(&tts->ort, trail_val);

        int trail_len = n_remaining + 1;  /* +1 for tts_eos */
        float *trail = (float *)malloc((size_t)trail_len * (size_t)H * sizeof(float));
        if (!trail) { tts_ort_release_value(&tts->ort, trail_val); goto cleanup; }

        memcpy(trail, trail_embed, (size_t)n_remaining * (size_t)H * sizeof(float));
        memcpy(trail + n_remaining * H, tts_eos_embed, (size_t)H * sizeof(float));

        tts_ort_release_value(&tts->ort, trail_val);
        *trailing_data = trail;
        *trailing_seq_len = trail_len;
    } else {
        /* No remaining text tokens, just tts_eos */
        float *trail = (float *)malloc((size_t)H * sizeof(float));
        if (!trail) goto cleanup;
        memcpy(trail, tts_eos_embed, (size_t)H * sizeof(float));
        *trailing_data = trail;
        *trailing_seq_len = 1;
    }

    tts_ort_release_value(&tts->ort, tts_special_val);
    tts_ort_release_value(&tts->ort, role_val);
    tts_ort_release_value(&tts->ort, codec_val);
    tts_ort_release_value(&tts->ort, text_first_val);
    return 0;

cleanup:
    tts_ort_release_value(&tts->ort, tts_special_val);
    tts_ort_release_value(&tts->ort, role_val);
    tts_ort_release_value(&tts->ort, codec_val);
    tts_ort_release_value(&tts->ort, text_first_val);
    free(*prefill_data); *prefill_data = NULL;
    free(*trailing_data); *trailing_data = NULL;
    free(*tts_pad_embed_out); *tts_pad_embed_out = NULL;
    return -1;
}

/* ---- Autoregressive decode loop ---- */

/* Run the autoregressive decode loop.
 * Returns malloc'd codes array [n_steps * NUM_CODE_GROUPS], sets *out_n_steps. */
static int64_t *run_decode_loop(TtsPipeline *tts,
                                float *prefill_embed, int prefill_seq_len,
                                float *trailing_embed, int trailing_seq_len,
                                float *tts_pad_embed,
                                float temperature, int top_k,
                                int *out_n_steps) {
    TtsOrt *ort = &tts->ort;
    int H = HIDDEN_SIZE;
    *out_n_steps = 0;

    /* Build attention mask (all ones for prefill) */
    int64_t *attn_mask = (int64_t *)malloc((size_t)(prefill_seq_len + MAX_DECODE_STEPS) * sizeof(int64_t));
    if (!attn_mask) return NULL;
    for (int i = 0; i < prefill_seq_len; i++) attn_mask[i] = 1;
    int attn_len = prefill_seq_len;

    /* Allocate codes output */
    int64_t *all_codes = (int64_t *)malloc((size_t)MAX_DECODE_STEPS * NUM_CODE_GROUPS * sizeof(int64_t));
    if (!all_codes) { free(attn_mask); return NULL; }

    /* History for repetition penalty */
    int *first_code_history = (int *)malloc((size_t)MAX_DECODE_STEPS * sizeof(int));
    if (!first_code_history) { free(attn_mask); free(all_codes); return NULL; }
    int history_len = 0;

    /* Suppress tokens: range [vocab_size-1024, vocab_size) except eos */
    int suppress_start = TALKER_VOCAB_SIZE - 1024;  /* 2048 */

    /* ---- Prefill ---- */
    if (tts->verbose) printf("  TTS prefill: seq_len=%d\n", prefill_seq_len);

    OrtValue *prefill_embed_val = tts_ort_tensor_float_3d(
        ort, prefill_embed, 1, (int64_t)prefill_seq_len, HIDDEN_SIZE);
    OrtValue *prefill_mask_val = tts_ort_tensor_int64_2d(
        ort, attn_mask, (int64_t)attn_len);

    if (!prefill_embed_val || !prefill_mask_val) {
        tts_ort_release_value(ort, prefill_embed_val);
        tts_ort_release_value(ort, prefill_mask_val);
        free(attn_mask); free(all_codes); free(first_code_history);
        return NULL;
    }

    /* Get prefill output names dynamically */
    OrtAllocator *allocator;
    ort->api->GetAllocatorWithDefaultOptions(&allocator);

    size_t n_prefill_outputs;
    ort->api->SessionGetOutputCount(ort->talker_prefill, &n_prefill_outputs);

    char **prefill_out_names = (char **)calloc(n_prefill_outputs, sizeof(char *));
    OrtValue **prefill_outputs = (OrtValue **)calloc(n_prefill_outputs, sizeof(OrtValue *));

    for (size_t i = 0; i < n_prefill_outputs; i++) {
        ort->api->SessionGetOutputName(ort->talker_prefill, i, allocator, &prefill_out_names[i]);
    }

    const char *prefill_in_names[] = { "inputs_embeds", "attention_mask" };
    const OrtValue *prefill_inputs[] = { prefill_embed_val, prefill_mask_val };

    int rc = tts_ort_run(ort, ort->talker_prefill,
                          prefill_in_names, prefill_inputs, 2,
                          (const char **)prefill_out_names, prefill_outputs, n_prefill_outputs);

    tts_ort_release_value(ort, prefill_embed_val);
    tts_ort_release_value(ort, prefill_mask_val);

    if (rc != 0) {
        for (size_t i = 0; i < n_prefill_outputs; i++) {
            ort->api->AllocatorFree(allocator, prefill_out_names[i]);
            tts_ort_release_value(ort, prefill_outputs[i]);
        }
        free(prefill_out_names); free(prefill_outputs);
        free(attn_mask); free(all_codes); free(first_code_history);
        return NULL;
    }

    /* Extract logits and last_hidden from prefill */
    OrtValue *logits_val = prefill_outputs[0];
    OrtValue *last_hidden_val = prefill_outputs[1];

    /* Transfer ownership out of prefill_outputs to prevent double-free.
     * logits_val, last_hidden_val, and kv_cache now own these tensors. */
    prefill_outputs[0] = NULL;
    prefill_outputs[1] = NULL;

    /* Store KV-cache tensors (outputs[2:]) */
    int n_kv = (int)n_prefill_outputs - 2;
    OrtValue **kv_cache = (OrtValue **)calloc((size_t)n_kv, sizeof(OrtValue *));
    for (int i = 0; i < n_kv; i++) {
        kv_cache[i] = prefill_outputs[i + 2];
        prefill_outputs[i + 2] = NULL;  /* prevent double-free */
    }

    /* Get decode session input/output names */
    size_t n_decode_inputs, n_decode_outputs;
    ort->api->SessionGetInputCount(ort->talker_decode, &n_decode_inputs);
    ort->api->SessionGetOutputCount(ort->talker_decode, &n_decode_outputs);

    char **decode_in_names = (char **)calloc(n_decode_inputs, sizeof(char *));
    char **decode_out_names = (char **)calloc(n_decode_outputs, sizeof(char *));
    for (size_t i = 0; i < n_decode_inputs; i++)
        ort->api->SessionGetInputName(ort->talker_decode, i, allocator, &decode_in_names[i]);
    for (size_t i = 0; i < n_decode_outputs; i++)
        ort->api->SessionGetOutputName(ort->talker_decode, i, allocator, &decode_out_names[i]);

    /* ---- Decode loop ---- */
    int n_steps = 0;
    int use_kv_cache = (n_kv > 0 && (int)n_decode_inputs > 2);

    for (int step = 0; step < MAX_DECODE_STEPS; step++) {
        /* Get logits from last position */
        float *logits_data = tts_ort_get_float_data(ort, logits_val);
        int64_t logits_shape[4];
        tts_ort_get_shape(ort, logits_val, logits_shape, 4);
        int seq_dim = (int)logits_shape[1];
        int vocab = (int)logits_shape[2];

        /* Get logits for last position */
        float *last_logits = logits_data + (seq_dim - 1) * vocab;

        /* Apply suppress tokens: suppress [2048..3071] except eos=2150 */
        for (int i = suppress_start; i < vocab; i++) {
            if (i != CODEC_EOS_TOKEN_ID) {
                last_logits[i] = -1.0e9f;
            }
        }

        /* Apply repetition penalty */
        tts_apply_repetition_penalty(last_logits, vocab,
                                     first_code_history, history_len, 1.05f);

        /* Sample first codebook token */
        int cb0 = tts_sample_topk(last_logits, vocab, temperature, top_k);

        if (cb0 == CODEC_EOS_TOKEN_ID) {
            if (tts->verbose) printf("  TTS: EOS at step %d\n", step);
            break;
        }

        first_code_history[history_len++] = cb0;

        /* Get last_hidden for this step */
        float *hidden_data = tts_ort_get_float_data(ort, last_hidden_val);
        int64_t hidden_shape[4];
        tts_ort_get_shape(ort, last_hidden_val, hidden_shape, 4);
        int hidden_seq = (int)hidden_shape[1];
        float *last_hidden = hidden_data + (hidden_seq - 1) * H;

        /* Get first codebook embedding: codec_embed(cb0) */
        int64_t cb0_id = (int64_t)cb0;
        OrtValue *first_embed_val = run_codec_embed(tts, &cb0_id, 1);
        if (!first_embed_val) break;
        float *first_embed = tts_ort_get_float_data(ort, first_embed_val);

        /* Store step codes */
        int64_t *step_codes = &all_codes[n_steps * NUM_CODE_GROUPS];
        step_codes[0] = (int64_t)cb0;

        /* Sub-code loop: predict codebooks 1-15 */
        /* Build embedding sequence incrementally: [last_hidden, first_embed, sub_embeds...] */
        int sub_seq_len = 2;  /* starts with last_hidden + first_embed */
        float *sub_embeds_buf = (float *)malloc((size_t)(2 + NUM_CODE_GROUPS) * (size_t)H * sizeof(float));
        if (!sub_embeds_buf) { tts_ort_release_value(ort, first_embed_val); break; }

        /* Copy last_hidden and first_embed */
        memcpy(sub_embeds_buf, last_hidden, (size_t)H * sizeof(float));
        memcpy(sub_embeds_buf + H, first_embed, (size_t)H * sizeof(float));

        /* Accumulate sum of sub-embeddings for codec_sum */
        float *codec_sum = (float *)calloc((size_t)H, sizeof(float));
        if (!codec_sum) {
            free(sub_embeds_buf);
            tts_ort_release_value(ort, first_embed_val);
            break;
        }

        /* Start codec_sum with first_embed */
        memcpy(codec_sum, first_embed, (size_t)H * sizeof(float));

        for (int j = 0; j < NUM_CODE_GROUPS - 1; j++) {
            /* Run code_predictor: inputs_embeds [1, sub_seq_len, H], generation_step [1] */
            OrtValue *sub_input_val = tts_ort_tensor_float_3d(
                ort, sub_embeds_buf, 1, (int64_t)sub_seq_len, HIDDEN_SIZE);
            int64_t gen_step = (int64_t)j;
            OrtValue *gen_step_val = tts_ort_tensor_int64(ort, &gen_step, 1);

            if (!sub_input_val || !gen_step_val) {
                tts_ort_release_value(ort, sub_input_val);
                tts_ort_release_value(ort, gen_step_val);
                break;
            }

            const char *cp_in_names[] = { tts->ort.code_pred_in_embeds, tts->ort.code_pred_in_step };
            const OrtValue *cp_inputs[] = { sub_input_val, gen_step_val };
            const char *cp_out_names[] = { tts->ort.code_pred_out };
            OrtValue *sub_logits_val = NULL;

            rc = tts_ort_run(ort, ort->code_predictor,
                              cp_in_names, cp_inputs, 2,
                              cp_out_names, &sub_logits_val, 1);

            tts_ort_release_value(ort, sub_input_val);
            tts_ort_release_value(ort, gen_step_val);

            if (rc != 0) break;

            /* Sample sub-code token */
            float *sub_logits = tts_ort_get_float_data(ort, sub_logits_val);
            int64_t sub_shape[4];
            tts_ort_get_shape(ort, sub_logits_val, sub_shape, 4);
            int sub_vocab = (int)sub_shape[sub_shape[0] > 1 ? 1 : (sub_shape[1] > 1 ? 2 : 1)];

            /* The sub-code predictor has vocab_size=2048 (from code_predictor_config) */
            sub_vocab = 2048;
            int sub_tok = tts_sample_topk(sub_logits, sub_vocab, temperature, top_k);
            tts_ort_release_value(ort, sub_logits_val);

            step_codes[j + 1] = (int64_t)sub_tok;

            /* Get sub-code embedding: code_predictor_embed(sub_tok, j) */
            int64_t sub_tok_id = (int64_t)sub_tok;
            OrtValue *sub_tok_input = tts_ort_tensor_int64_2d(ort, &sub_tok_id, 1);
            int64_t gen_step_j = (int64_t)j;
            OrtValue *gen_step_j_val = tts_ort_tensor_int64(ort, &gen_step_j, 1);

            if (!sub_tok_input || !gen_step_j_val) {
                tts_ort_release_value(ort, sub_tok_input);
                tts_ort_release_value(ort, gen_step_j_val);
                break;
            }

            const char *cpe_in_names[] = { tts->ort.code_pred_embed_in_ids, tts->ort.code_pred_embed_in_step };
            const OrtValue *cpe_inputs[] = { sub_tok_input, gen_step_j_val };
            const char *cpe_out_names[] = { tts->ort.code_pred_embed_out };
            OrtValue *sub_embed_val = NULL;

            rc = tts_ort_run(ort, ort->code_predictor_embed,
                              cpe_in_names, cpe_inputs, 2,
                              cpe_out_names, &sub_embed_val, 1);

            tts_ort_release_value(ort, sub_tok_input);
            tts_ort_release_value(ort, gen_step_j_val);

            if (rc != 0) break;

            float *sub_embed = tts_ort_get_float_data(ort, sub_embed_val);

            /* Add to embed sequence for next sub-code prediction */
            memcpy(sub_embeds_buf + sub_seq_len * H, sub_embed, (size_t)H * sizeof(float));
            sub_seq_len++;

            /* Accumulate into codec_sum */
            for (int k = 0; k < H; k++) {
                codec_sum[k] += sub_embed[k];
            }

            tts_ort_release_value(ort, sub_embed_val);
        }

        tts_ort_release_value(ort, first_embed_val);

        /* Add trailing text hidden or tts_pad_embed */
        if (step < trailing_seq_len) {
            for (int k = 0; k < H; k++) {
                codec_sum[k] += trailing_embed[step * H + k];
            }
        } else {
            for (int k = 0; k < H; k++) {
                codec_sum[k] += tts_pad_embed[k];
            }
        }

        n_steps++;

        /* Prepare for next step */
        /* Extend attention mask */
        attn_mask[attn_len++] = 1;

        /* Release previous logits/hidden */
        tts_ort_release_value(ort, logits_val);
        tts_ort_release_value(ort, last_hidden_val);
        logits_val = NULL;
        last_hidden_val = NULL;

        if (use_kv_cache) {
            /* Decode step with KV cache */
            OrtValue *step_embed_val = tts_ort_tensor_float_3d(
                ort, codec_sum, 1, 1, HIDDEN_SIZE);
            OrtValue *step_mask_val = tts_ort_tensor_int64_2d(
                ort, attn_mask, (int64_t)attn_len);

            if (!step_embed_val || !step_mask_val) {
                tts_ort_release_value(ort, step_embed_val);
                tts_ort_release_value(ort, step_mask_val);
                free(sub_embeds_buf); free(codec_sum);
                break;
            }

            /* Build decode inputs: [embed, mask, kv_cache...] */
            size_t n_dec_in = 2 + (size_t)n_kv;
            const OrtValue **dec_inputs = (const OrtValue **)malloc(n_dec_in * sizeof(OrtValue *));
            dec_inputs[0] = step_embed_val;
            dec_inputs[1] = step_mask_val;
            for (int i = 0; i < n_kv; i++) {
                dec_inputs[2 + i] = kv_cache[i];
            }

            OrtValue **dec_outputs = (OrtValue **)calloc(n_decode_outputs, sizeof(OrtValue *));

            rc = tts_ort_run(ort, ort->talker_decode,
                              (const char **)decode_in_names, dec_inputs, n_dec_in,
                              (const char **)decode_out_names, dec_outputs, n_decode_outputs);

            tts_ort_release_value(ort, step_embed_val);
            tts_ort_release_value(ort, step_mask_val);
            free((void *)dec_inputs);

            if (rc != 0) {
                free(dec_outputs);
                free(sub_embeds_buf); free(codec_sum);
                break;
            }

            logits_val = dec_outputs[0];
            last_hidden_val = dec_outputs[1];

            /* Update KV cache */
            for (int i = 0; i < n_kv; i++) {
                tts_ort_release_value(ort, kv_cache[i]);
                kv_cache[i] = dec_outputs[2 + i];
            }

            free(dec_outputs);
        } else {
            /* No KV cache -- re-run prefill with full sequence (slow fallback) */
            /* This shouldn't happen with properly exported models */
            fprintf(stderr, "TTS: WARNING: no KV cache support, using slow fallback\n");
            free(sub_embeds_buf); free(codec_sum);
            break;
        }

        free(sub_embeds_buf);
        free(codec_sum);

        if (tts->verbose && (step % 50 == 0 || step < 5)) {
            printf("  TTS decode step %d: cb0=%d\n", step, cb0);
        }
    }

    /* Cleanup */
    tts_ort_release_value(ort, logits_val);
    tts_ort_release_value(ort, last_hidden_val);
    for (int i = 0; i < n_kv; i++) {
        tts_ort_release_value(ort, kv_cache[i]);
    }
    free(kv_cache);

    for (size_t i = 0; i < n_prefill_outputs; i++) {
        ort->api->AllocatorFree(allocator, prefill_out_names[i]);
        tts_ort_release_value(ort, prefill_outputs[i]);
    }
    free(prefill_out_names);
    free(prefill_outputs);

    for (size_t i = 0; i < n_decode_inputs; i++)
        ort->api->AllocatorFree(allocator, decode_in_names[i]);
    for (size_t i = 0; i < n_decode_outputs; i++)
        ort->api->AllocatorFree(allocator, decode_out_names[i]);
    free(decode_in_names);
    free(decode_out_names);

    free(attn_mask);
    free(first_code_history);

    *out_n_steps = n_steps;
    return all_codes;
}

/* ---- Vocoder decode ---- */

/* Run tokenizer12hz_decode: audio_codes [1, n_steps, 16] -> audio [1, n_samples] */
static float *run_vocoder(TtsPipeline *tts, const int64_t *codes, int n_steps,
                          int *out_n_samples) {
    /* Reshape codes to [1, n_steps, 16] */
    OrtValue *codes_input = NULL;
    int64_t shape[] = { 1, (int64_t)n_steps, NUM_CODE_GROUPS };
    size_t byte_len = (size_t)n_steps * NUM_CODE_GROUPS * sizeof(int64_t);
    OrtStatus *status = tts->ort.api->CreateTensorWithDataAsOrtValue(
        tts->ort.mem_info, (void *)codes, byte_len,
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &codes_input);
    if (status) {
        fprintf(stderr, "TTS vocoder: tensor error: %s\n", tts->ort.api->GetErrorMessage(status));
        tts->ort.api->ReleaseStatus(status);
        return NULL;
    }

    const char *in_names[] = { tts->ort.vocoder_in };
    /* tokenizer12hz_decode outputs: audio_values [1, n_samples], out_lengths [1] */
    int n_voc_out = tts->ort.vocoder_out_lengths ? 2 : 1;
    const char *out_names[2];
    out_names[0] = tts->ort.vocoder_out_audio;
    out_names[1] = tts->ort.vocoder_out_lengths;
    OrtValue *outputs[2] = { NULL, NULL };

    if (tts->verbose) {
        printf("  vocoder: running ONNX inference (input shape [1, %d, %d])...\n",
               n_steps, NUM_CODE_GROUPS);
        fflush(stdout);
    }
    int rc = tts_ort_run(&tts->ort, tts->ort.tokenizer12hz_decode,
                          in_names, (const OrtValue *const *)&codes_input, 1,
                          out_names, outputs, (size_t)n_voc_out);
    tts_ort_release_value(&tts->ort, codes_input);

    if (rc != 0) return NULL;
    if (tts->verbose) {
        printf("  vocoder: ONNX inference completed\n");
        fflush(stdout);
    }

    /* Get audio data */
    float *audio = tts_ort_get_float_data(&tts->ort, outputs[0]);
    int64_t audio_shape[4];
    int ndims = tts_ort_get_shape(&tts->ort, outputs[0], audio_shape, 4);

    int n_samples;
    if (outputs[1]) {
        int64_t *lengths = tts_ort_get_int64_data(&tts->ort, outputs[1]);
        n_samples = lengths ? (int)lengths[0] : 0;
    } else {
        n_samples = (ndims >= 2) ? (int)audio_shape[1] : (int)audio_shape[0];
    }

    /* Clamp to actual tensor size */
    int64_t total_elements = 1;
    for (int i = 0; i < ndims; i++) total_elements *= audio_shape[i];
    if (n_samples > (int)total_elements) n_samples = (int)total_elements;
    if (n_samples <= 0) {
        /* Fallback: use shape-based length */
        n_samples = (ndims >= 2) ? (int)audio_shape[1] : (int)audio_shape[0];
    }

    /* Copy audio data (tensor will be freed) */
    float *result = (float *)malloc((size_t)n_samples * sizeof(float));
    if (result) {
        memcpy(result, audio, (size_t)n_samples * sizeof(float));
    }

    *out_n_samples = n_samples;
    tts_ort_release_value(&tts->ort, outputs[0]);
    tts_ort_release_value(&tts->ort, outputs[1]);
    return result;
}

/* ---- Main synthesis function ---- */

int tts_pipeline_synthesize(TtsPipeline *tts, const char *text,
                            float temperature, int top_k, float speed,
                            TtsResult *result) {
    (void)speed;  /* not implemented yet */
    memset(result, 0, sizeof(*result));

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    if (tts->verbose) {
        printf("TTS synthesize: \"%s\"\n", text);
    }

    /* 1. Build input token sequence */
    int n_ids = 0, role_len = 0;
    int64_t *input_ids = build_input_ids(tts, text, &n_ids, &role_len);
    if (!input_ids) return -1;

    if (tts->verbose) {
        printf("  tokens: %d total, %d role, %d text, 5 end\n",
               n_ids, role_len, n_ids - role_len - 5);
    }

    /* 2. Build prefill embeddings */
    float *prefill_data = NULL, *trailing_data = NULL, *tts_pad_embed = NULL;
    int prefill_seq_len = 0, trailing_seq_len = 0;

    int rc = build_prefill_embeddings(tts, input_ids, n_ids, role_len,
                                       &prefill_data, &prefill_seq_len,
                                       &trailing_data, &trailing_seq_len,
                                       &tts_pad_embed);
    free(input_ids);
    if (rc != 0) return -1;

    if (tts->verbose) {
        printf("  prefill seq_len=%d, trailing=%d\n", prefill_seq_len, trailing_seq_len);
    }

    /* 3. Autoregressive decode loop */
    int n_steps = 0;
    int64_t *codes = run_decode_loop(tts, prefill_data, prefill_seq_len,
                                      trailing_data, trailing_seq_len,
                                      tts_pad_embed, temperature, top_k,
                                      &n_steps);
    free(prefill_data);
    free(trailing_data);
    free(tts_pad_embed);

    if (!codes || n_steps == 0) {
        free(codes);
        fprintf(stderr, "TTS: decode produced no steps\n");
        return -1;
    }

    if (tts->verbose) {
        printf("  decode: %d steps (%.1f seconds of audio)\n",
               n_steps, n_steps * 0.08);
    }

    /* 4. Vocoder decode: codes -> audio */
    if (tts->verbose) {
        printf("  vocoder: starting (%d steps, %d code groups)...\n",
               n_steps, NUM_CODE_GROUPS);
        fflush(stdout);
    }
    int n_samples = 0;
    float *audio = run_vocoder(tts, codes, n_steps, &n_samples);
    free(codes);

    if (!audio || n_samples == 0) {
        free(audio);
        fprintf(stderr, "TTS: vocoder produced no audio\n");
        return -1;
    }

    if (tts->verbose) {
        printf("  vocoder: %d samples (%.1f seconds at %d Hz)\n",
               n_samples, (double)n_samples / WAV_SAMPLE_RATE, WAV_SAMPLE_RATE);
    }

    /* 5. Encode as WAV */
    size_t wav_len = 0;
    unsigned char *wav = encode_wav(audio, n_samples, &wav_len);
    free(audio);

    if (!wav) {
        fprintf(stderr, "TTS: WAV encoding failed\n");
        return -1;
    }

    QueryPerformanceCounter(&t_end);
    double elapsed_ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;

    result->wav_data = wav;
    result->wav_len = wav_len;
    result->n_samples = n_samples;
    result->n_steps = n_steps;
    result->elapsed_ms = elapsed_ms;

    if (tts->verbose) {
        printf("  TTS complete: %d steps, %d samples, %.0f ms\n",
               n_steps, n_samples, elapsed_ms);
    }

    return 0;
}

#endif /* USE_ORT */
