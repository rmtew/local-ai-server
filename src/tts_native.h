/*
 * tts_native.h - Native C+cuBLAS TTS inference for Qwen3-TTS
 *
 * Reuses qwen-asr infrastructure (safetensors, tokenizer, kernels, GPU)
 * for the talker LM and code predictor. Vocoder is in tts_vocoder.c.
 *
 * Supports 0.6B (talker_hidden=1024) and 1.7B (talker_hidden=2048).
 * Model size is auto-detected from weight shapes at init time.
 *
 * Architecture:
 *   Talker LM: 28-layer Qwen3 transformer (hidden=1024 or 2048)
 *   Code Predictor: 5-layer Qwen3 transformer (hidden=1024, always)
 *   Embeddings: text (151936x2048 -> MLP -> talker_hidden), codec (3072xtalker_hidden)
 */

#ifndef LOCAL_AI_TTS_NATIVE_H
#define LOCAL_AI_TTS_NATIVE_H

#include "qwen_asr.h"           /* qwen_config_t, qwen_dec_layer_t, etc. */
#include "qwen_asr_safetensors.h"
#include "qwen_asr_tokenizer.h"
#include "tts_speaker_enc.h"
#include <stdint.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define TTS_TALKER_LAYERS       28
#define TTS_CODEPRED_LAYERS     5
#define TTS_TEXT_HIDDEN_SIZE    2048     /* text embedding dim (same for all models) */
#define TTS_NUM_HEADS           16
#define TTS_NUM_KV_HEADS        8
#define TTS_HEAD_DIM            128
#define TTS_TALKER_VOCAB        3072
#define TTS_CODEC_VOCAB         2048
#define TTS_TEXT_VOCAB          151936
#define TTS_NUM_CODE_GROUPS     16
#define TTS_MAX_DECODE_STEPS    200     /* default, overridable via ctx->max_steps */
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
 * Text Projection MLP: text_embed(2048) -> SiLU -> talker_hidden
 * ======================================================================== */

typedef struct {
    uint16_t *fc1_weight_bf16;  /* [intermediate, 2048] */
    float *fc1_bias;            /* [intermediate] or NULL */
    uint16_t *fc2_weight_bf16;  /* [talker_hidden, intermediate] */
    float *fc2_bias;            /* [talker_hidden] or NULL */
    int intermediate;           /* determined from weight shape */
} tts_text_project_t;

/* ========================================================================
 * Code Predictor (5-layer transformer, no persistent KV cache)
 * ======================================================================== */

typedef struct {
    qwen_dec_layer_t layers[TTS_CODEPRED_LAYERS];
    float *norm;                                        /* [cp_hidden] final RMSNorm */

    /* Per-codebook embeddings and lm_heads (indices 0..14 for sub-codebooks 1..15).
     * codec_embed is in TALKER space (for codec_sum feedback), lm_head is in CP space. */
    uint16_t *codec_embed_bf16[TTS_NUM_CODE_GROUPS - 1]; /* [2048, talker_hidden] each */
    uint16_t *lm_head_bf16[TTS_NUM_CODE_GROUPS - 1];     /* [2048, cp_hidden] each */

    /* Projection from talker hidden to code predictor hidden.
     * NULL when talker_hidden == cp_hidden (0.6B model). */
    float *mtp_proj_weight;     /* [cp_hidden, talker_hidden] or NULL */
    float *mtp_proj_bias;       /* [cp_hidden] or NULL */
} tts_code_predictor_t;

/* ========================================================================
 * Main TTS Context
 * ======================================================================== */

typedef struct {
    /* Talker LM (reuses qwen_decoder_t for layers + norm) */
    qwen_config_t talker_config;    /* dec_hidden=1024(0.6B) or 2048(1.7B) */
    qwen_decoder_t talker;

    /* Code predictor config (hidden=1024 for all models) */
    qwen_config_t cp_config;

    /* Separate lm_head (talker.codec_head, NOT tied with embeddings) */
    uint16_t *codec_head_bf16;  /* [3072, talker_hidden] */

    /* Text embedding + projection */
    uint16_t *text_embed_bf16;  /* [151936, 2048] */
    tts_text_project_t text_project;

    /* Codec embedding (talker.model.codec_embedding) */
    uint16_t *codec_embed_bf16; /* [3072, talker_hidden] */

    /* Code predictor */
    tts_code_predictor_t code_pred;

    /* Tokenizer */
    qwen_tokenizer_t *tokenizer;

    /* Speaker encoder (optional, Base model only) */
    tts_speaker_enc_ctx_t speaker_enc;
    int spk_embed_dim;              /* 1024(0.6B) or 2048(1.7B), = talker_hidden */

    /* Model files (kept open for mmap) */
    multi_safetensors_t *safetensors;

    /* KV cache for talker */
    float *kv_cache_k;          /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent talker decode buffers (single-token, sized for talker_hidden) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_ffn_out;

    /* Prefill buffers (also reused for code predictor forward pass).
     * Allocated at talker dimensions (>= code predictor dimensions). */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;

    /* Code predictor input embedding buffer */
    float *cp_embed_buf;        /* [18, cp_hidden] for [projected_hidden, cb0_embed, sub_embeds...] */

    /* Decode loop accumulation buffer */
    float *codec_sum_buf;       /* [talker_hidden] codec_sum for talker feedback */

    /* RoPE cache (shared for talker and code predictor -- same theta) */
    float *rope_cache_cos, *rope_cache_sin;
    float *rope_inv_freq;
    int rope_cache_cap;

    /* Logits buffer for sampling (talker) */
    float *logits_buf;          /* [TTS_TALKER_VOCAB] */

    /* Logits buffer for sampling (code predictor) */
    float *cp_logits_buf;       /* [TTS_CODEC_VOCAB] */

    /* Temporary embedding buffers */
    float *tmp_embed;           /* [talker_hidden] scratch for embeddings */
    float *tmp_text_embed;      /* [TTS_TEXT_HIDDEN_SIZE] for text projection */

    /* GPU context (from qwen-asr, shared) */
#ifdef USE_CUBLAS
    void *gpu_ctx;              /* qwen_gpu_ctx_t* -- shared with ASR if both loaded */
#endif

    int max_steps;              /* max decode steps (default TTS_MAX_DECODE_STEPS) */
    int verbose;
} tts_native_ctx_t;

/* ========================================================================
 * API
 * ======================================================================== */

/* Load the native TTS model from a directory containing model.safetensors
 * and tokenizer files (vocab.json, merges.txt).
 * gpu_ctx may be NULL (CPU-only) or a qwen_gpu_ctx_t* for GPU acceleration.
 * fp16: if non-zero, store GPU weights as FP16 (half VRAM, uses tensor cores).
 * Returns 0 on success. */
int tts_native_init(tts_native_ctx_t *ctx, const char *model_dir,
                    void *gpu_ctx, int fp16, int verbose);

/* Free all resources. */
void tts_native_free(tts_native_ctx_t *ctx);

/* Run talker + code predictor to produce codec tokens.
 * language: language string ("auto", "english", "chinese", etc.) or NULL for auto.
 * speaker_embed: optional speaker embedding (NULL for default voice).
 *                Must be spk_embed_dim floats (1024 for 0.6B, 2048 for 1.7B).
 * Returns 0 on success. Caller must free *codes_out.
 * codes_out: [n_steps * 16] int64 array of codec tokens
 * n_steps_out: number of decode steps */
int tts_native_decode(tts_native_ctx_t *ctx, const char *text,
                      const char *language, const float *speaker_embed,
                      float temperature, int top_k,
                      int64_t **codes_out, int *n_steps_out);

#endif /* LOCAL_AI_TTS_NATIVE_H */
