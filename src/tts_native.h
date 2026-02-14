/*
 * tts_native.h - Native C+cuBLAS TTS inference for Qwen3-TTS
 *
 * Reuses qwen-asr infrastructure (safetensors, tokenizer, kernels, GPU)
 * for the talker LM and code predictor. Vocoder remains in ONNX for now.
 *
 * Architecture:
 *   Talker LM: 28-layer Qwen3 transformer (identical to qwen-asr decoder)
 *   Code Predictor: 5-layer Qwen3 transformer (same arch, smaller)
 *   Embeddings: text (151936x2048 -> MLP -> 1024), codec (3072x1024)
 */

#ifndef LOCAL_AI_TTS_NATIVE_H
#define LOCAL_AI_TTS_NATIVE_H

#include "qwen_asr.h"           /* qwen_config_t, qwen_dec_layer_t, etc. */
#include "qwen_asr_safetensors.h"
#include "qwen_asr_tokenizer.h"
#include <stdint.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define TTS_TALKER_LAYERS       28
#define TTS_CODEPRED_LAYERS     5
#define TTS_HIDDEN_SIZE         1024
#define TTS_TEXT_HIDDEN_SIZE    2048
#define TTS_NUM_HEADS           16
#define TTS_NUM_KV_HEADS        8
#define TTS_HEAD_DIM            128
#define TTS_INTERMEDIATE        3072
#define TTS_TALKER_VOCAB        3072
#define TTS_CODEC_VOCAB         2048
#define TTS_TEXT_VOCAB          151936
#define TTS_NUM_CODE_GROUPS     16
#define TTS_MAX_DECODE_STEPS    50
#define TTS_WAV_SAMPLE_RATE     24000
#define TTS_ROPE_THETA          1000000.0f
#define TTS_RMS_NORM_EPS        1e-6f

/* Special token IDs -- text domain */
#define TTS_TOKEN_IM_START      151644
#define TTS_TOKEN_IM_END        151645
#define TTS_TOKEN_BOS           151672
#define TTS_TOKEN_EOS           151673
#define TTS_TOKEN_PAD           151671

/* Special token IDs -- codec domain */
#define TTS_CODEC_BOS           2149
#define TTS_CODEC_EOS           2150
#define TTS_CODEC_PAD           2148
#define TTS_CODEC_NOTHINK       2155
#define TTS_CODEC_THINK_BOS     2156
#define TTS_CODEC_THINK_EOS     2157

/* ========================================================================
 * Text Projection MLP: text_embed(2048) -> SiLU -> 1024
 * ======================================================================== */

typedef struct {
    uint16_t *fc1_weight_bf16;  /* [intermediate, 2048] */
    float *fc1_bias;            /* [intermediate] or NULL */
    uint16_t *fc2_weight_bf16;  /* [1024, intermediate] */
    float *fc2_bias;            /* [1024] or NULL */
    int intermediate;           /* determined from weight shape */
} tts_text_project_t;

/* ========================================================================
 * Code Predictor (5-layer transformer, no persistent KV cache)
 * ======================================================================== */

typedef struct {
    qwen_dec_layer_t layers[TTS_CODEPRED_LAYERS];
    float *norm;                                        /* [1024] final RMSNorm */

    /* Per-codebook embeddings and lm_heads (indices 0..14 for sub-codebooks 1..15) */
    uint16_t *codec_embed_bf16[TTS_NUM_CODE_GROUPS - 1]; /* [2048, 1024] each */
    uint16_t *lm_head_bf16[TTS_NUM_CODE_GROUPS - 1];     /* [2048, 1024] each */

    /* Projection from talker hidden to code predictor hidden */
    float *mtp_proj_weight;     /* [1024, 1024] */
    float *mtp_proj_bias;       /* [1024] or NULL */
} tts_code_predictor_t;

/* ========================================================================
 * Main TTS Context
 * ======================================================================== */

typedef struct {
    /* Talker LM (reuses qwen_decoder_t for layers + norm) */
    qwen_config_t talker_config;
    qwen_decoder_t talker;

    /* Separate lm_head (talker.codec_head, NOT tied with embeddings) */
    uint16_t *codec_head_bf16;  /* [3072, 1024] */

    /* Text embedding + projection */
    uint16_t *text_embed_bf16;  /* [151936, 2048] */
    tts_text_project_t text_project;

    /* Codec embedding (talker.model.codec_embedding) */
    uint16_t *codec_embed_bf16; /* [3072, 1024] */

    /* Code predictor */
    tts_code_predictor_t code_pred;

    /* Tokenizer */
    qwen_tokenizer_t *tokenizer;

    /* Model files (kept open for mmap) */
    multi_safetensors_t *safetensors;

    /* KV cache for talker */
    float *kv_cache_k;          /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent talker decode buffers (single-token) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_ffn_out;

    /* Prefill buffers (also reused for code predictor forward pass) */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;

    /* Code predictor input embedding buffer */
    float *cp_embed_buf;        /* [18, 1024] for [projected_hidden, cb0_embed, sub_embeds...] */

    /* RoPE cache (shared for talker and code predictor -- same theta) */
    float *rope_cache_cos, *rope_cache_sin;
    float *rope_inv_freq;
    int rope_cache_cap;

    /* Logits buffer for sampling (talker) */
    float *logits_buf;          /* [TTS_TALKER_VOCAB] */

    /* Logits buffer for sampling (code predictor) */
    float *cp_logits_buf;       /* [TTS_CODEC_VOCAB] */

    /* Temporary embedding buffers */
    float *tmp_embed;           /* [TTS_HIDDEN_SIZE] */
    float *tmp_text_embed;      /* [TTS_TEXT_HIDDEN_SIZE] for text projection */

    /* GPU context (from qwen-asr, shared) */
#ifdef USE_CUBLAS
    void *gpu_ctx;              /* qwen_gpu_ctx_t* -- shared with ASR if both loaded */
#endif

    int verbose;
} tts_native_ctx_t;

/* ========================================================================
 * API
 * ======================================================================== */

/* Load the native TTS model from a directory containing model.safetensors
 * and tokenizer files (vocab.json, merges.txt).
 * gpu_ctx may be NULL (CPU-only) or a qwen_gpu_ctx_t* for GPU acceleration.
 * Returns 0 on success. */
int tts_native_init(tts_native_ctx_t *ctx, const char *model_dir,
                    void *gpu_ctx, int verbose);

/* Free all resources. */
void tts_native_free(tts_native_ctx_t *ctx);

/* Run talker + code predictor to produce codec tokens.
 * Returns 0 on success. Caller must free *codes_out.
 * codes_out: [n_steps * 16] int64 array of codec tokens
 * n_steps_out: number of decode steps */
int tts_native_decode(tts_native_ctx_t *ctx, const char *text,
                      float temperature, int top_k,
                      int64_t **codes_out, int *n_steps_out);

#endif /* LOCAL_AI_TTS_NATIVE_H */
