/*
 * tts_native.c - Native C+cuBLAS TTS inference for Qwen3-TTS
 *
 * Implements the talker LM (28-layer Qwen3) and code predictor (5-layer)
 * using the same infrastructure as qwen-asr: safetensors mmap, CPU kernels
 * with AVX2/OpenBLAS, and optional cuBLAS GPU GEMM offload.
 *
 * The vocoder (codec tokens -> audio) is NOT implemented here; it remains
 * in ONNX Runtime via tts_pipeline.c for now.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_native.h"
#include "qwen_asr_kernels.h"
#ifdef USE_CUBLAS
#include "qwen_asr_gpu.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* From tts_sampling.c */
extern void tts_apply_repetition_penalty(float *logits, int vocab_size,
                                         const int *history, int history_len,
                                         float penalty);
extern int tts_sample_topk(float *logits, int vocab_size,
                           float temperature, int top_k);

/* ========================================================================
 * Helpers
 * ======================================================================== */

static void bf16_to_f32_row(float *dst, const uint16_t *src, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &bits, sizeof(float));
    }
}

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_bf16_direct(sf, t);
}

/* Try to load as f32 (silent on missing). */
static float *try_load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

/* Linear projection: tries GPU GEMM first, falls back to CPU.
 * out[M,N] = in[M,K] @ W[N,K]^T */
static void linear_bf16(tts_native_ctx_t *ctx, float *out, const float *in,
                        const uint16_t *weight_bf16, int M, int K, int N) {
#ifdef USE_CUBLAS
    qwen_gpu_ctx_t *gpu = (qwen_gpu_ctx_t *)ctx->gpu_ctx;
    if (gpu) {
        int h = qwen_gpu_find_weight(gpu, weight_bf16);
        if (h >= 0) {
            qwen_gpu_gemm(gpu, out, in, h, M, K, N);
            return;
        }
    }
#endif
    qwen_linear_nobias_bf16(out, in, weight_bf16, M, K, N);
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static int load_talker_weights(tts_native_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    qwen_decoder_t *dec = &ctx->talker;
    char name[512];

    /* Talker does NOT use tied embeddings -- tok_embeddings_bf16 unused */
    dec->tok_embeddings_bf16 = NULL;

    for (int i = 0; i < TTS_TALKER_LAYERS; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];

        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_proj.weight", i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_proj.weight", i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.v_proj.weight", i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.o_proj.weight", i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_norm.weight", i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_norm.weight", i);
        l->k_norm_weight = load_f32(ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.input_layernorm.weight", i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_f32(ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.gate_proj.weight", i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.up_proj.weight", i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.down_proj.weight", i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "TTS native: failed to load talker layer %d\n", i);
            return -1;
        }

        /* Fuse gate+up weights: interleave rows [gate_row0, up_row0, ...] */
        {
            int inter = TTS_INTERMEDIATE;
            int hidden = TTS_HIDDEN_SIZE;
            size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
            if (!l->gate_up_fused_bf16) return -1;
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                       l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                       l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
            }
        }
    }

    dec->norm = load_f32(ms, "talker.model.norm.weight");
    if (!dec->norm) {
        fprintf(stderr, "TTS native: failed to load talker final norm\n");
        return -1;
    }

    /* Codec head (separate lm_head, NOT tied) */
    ctx->codec_head_bf16 = load_bf16_direct(ms, "talker.codec_head.weight");
    if (!ctx->codec_head_bf16) {
        fprintf(stderr, "TTS native: failed to load codec_head\n");
        return -1;
    }

    return 0;
}

static int load_code_predictor_weights(tts_native_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    tts_code_predictor_t *cp = &ctx->code_pred;
    char name[512];

    for (int i = 0; i < TTS_CODEPRED_LAYERS; i++) {
        qwen_dec_layer_t *l = &cp->layers[i];

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_proj.weight", i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_proj.weight", i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.v_proj.weight", i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.o_proj.weight", i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_norm.weight", i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_norm.weight", i);
        l->k_norm_weight = load_f32(ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.input_layernorm.weight", i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_f32(ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.gate_proj.weight", i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.up_proj.weight", i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.down_proj.weight", i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "TTS native: failed to load code predictor layer %d\n", i);
            return -1;
        }

        /* Fuse gate+up */
        {
            int inter = TTS_INTERMEDIATE;
            int hidden = TTS_HIDDEN_SIZE;
            size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
            if (!l->gate_up_fused_bf16) return -1;
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                       l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                       l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
            }
        }
    }

    cp->norm = load_f32(ms, "talker.code_predictor.model.norm.weight");
    if (!cp->norm) {
        fprintf(stderr, "TTS native: failed to load code predictor norm\n");
        return -1;
    }

    /* Per-codebook embeddings and lm_heads (15 of each) */
    for (int j = 0; j < TTS_NUM_CODE_GROUPS - 1; j++) {
        snprintf(name, sizeof(name), "talker.code_predictor.model.codec_embedding.%d.weight", j);
        cp->codec_embed_bf16[j] = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.lm_head.%d.weight", j);
        cp->lm_head_bf16[j] = load_bf16_direct(ms, name);
        if (!cp->codec_embed_bf16[j] || !cp->lm_head_bf16[j]) {
            fprintf(stderr, "TTS native: failed to load code predictor codebook %d\n", j);
            return -1;
        }
    }

    /* MTP projection (talker hidden -> code predictor hidden).
     * Optional: when talker and code predictor have the same hidden size (0.6B),
     * this projection doesn't exist -- hidden state passes through directly. */
    cp->mtp_proj_weight = try_load_f32(ms, "talker.code_predictor.small_to_mtp_projection.weight");
    cp->mtp_proj_bias = try_load_f32(ms, "talker.code_predictor.small_to_mtp_projection.bias");
    if (ctx->verbose) {
        printf("TTS native: mtp_projection: %s\n",
               cp->mtp_proj_weight ? "loaded" : "none (same hidden size)");
    }

    return 0;
}

static int load_embedding_weights(tts_native_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;

    /* Text embedding: [151936, 2048] bf16 */
    ctx->text_embed_bf16 = load_bf16_direct(ms, "talker.model.text_embedding.weight");
    if (!ctx->text_embed_bf16) {
        fprintf(stderr, "TTS native: failed to load text_embedding\n");
        return -1;
    }

    /* Codec embedding: [3072, 1024] bf16 */
    ctx->codec_embed_bf16 = load_bf16_direct(ms, "talker.model.codec_embedding.weight");
    if (!ctx->codec_embed_bf16) {
        fprintf(stderr, "TTS native: failed to load codec_embedding\n");
        return -1;
    }

    /* Text projection MLP */
    tts_text_project_t *tp = &ctx->text_project;

    /* Determine intermediate size from fc1 weight shape */
    safetensors_file_t *sf = NULL;
    const safetensor_t *fc1_t = multi_safetensors_find(ms,
        "talker.text_projection.linear_fc1.weight", &sf);
    if (!fc1_t) {
        fprintf(stderr, "TTS native: failed to find text_projection.fc1\n");
        return -1;
    }
    tp->intermediate = (int)fc1_t->shape[0];

    tp->fc1_weight_bf16 = load_bf16_direct(ms, "talker.text_projection.linear_fc1.weight");
    tp->fc2_weight_bf16 = load_bf16_direct(ms, "talker.text_projection.linear_fc2.weight");
    if (!tp->fc1_weight_bf16 || !tp->fc2_weight_bf16) {
        fprintf(stderr, "TTS native: failed to load text_projection weights\n");
        return -1;
    }

    /* Biases are optional */
    tp->fc1_bias = try_load_f32(ms, "talker.text_projection.linear_fc1.bias");
    tp->fc2_bias = try_load_f32(ms, "talker.text_projection.linear_fc2.bias");

    if (ctx->verbose) {
        printf("TTS native: text_projection intermediate=%d, fc1_bias=%s, fc2_bias=%s\n",
               tp->intermediate,
               tp->fc1_bias ? "yes" : "no",
               tp->fc2_bias ? "yes" : "no");
    }

    return 0;
}

/* ========================================================================
 * KV Cache Management (Talker)
 * ======================================================================== */

static inline float *kv_k_at(tts_native_ctx_t *ctx, int layer, int pos) {
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static inline float *kv_v_at(tts_native_ctx_t *ctx, int layer, int pos) {
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static int kv_cache_init(tts_native_ctx_t *ctx, int max_seq) {
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    size_t size = (size_t)TTS_TALKER_LAYERS * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, size);
    ctx->kv_cache_v = (float *)calloc(1, size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    return (!ctx->kv_cache_k || !ctx->kv_cache_v) ? -1 : 0;
}

static int kv_cache_grow(tts_native_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)TTS_TALKER_LAYERS * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    for (int i = 0; i < TTS_TALKER_LAYERS; i++) {
        memcpy(new_k + i * new_stride, ctx->kv_cache_k + i * old_stride,
               (size_t)ctx->kv_cache_len * kv_dim * sizeof(float));
        memcpy(new_v + i * new_stride, ctx->kv_cache_v + i * old_stride,
               (size_t)ctx->kv_cache_len * kv_dim * sizeof(float));
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

/* ========================================================================
 * RoPE Cache
 * ======================================================================== */

static int ensure_rope_cache(tts_native_ctx_t *ctx, int n_pos) {
    int head_dim = TTS_HEAD_DIM;
    int half = head_dim / 2;

    /* Lazy init inv_freq */
    if (!ctx->rope_inv_freq) {
        ctx->rope_inv_freq = (float *)malloc((size_t)half * sizeof(float));
        if (!ctx->rope_inv_freq) return -1;
        for (int d = 0; d < half; d++) {
            ctx->rope_inv_freq[d] = 1.0f / powf(TTS_ROPE_THETA, (float)(2 * d) / (float)head_dim);
        }
    }

    if (n_pos <= ctx->rope_cache_cap) return 0;

    int new_cap = ctx->rope_cache_cap ? ctx->rope_cache_cap : 256;
    while (new_cap < n_pos) new_cap *= 2;

    float *new_cos = (float *)realloc(ctx->rope_cache_cos, (size_t)new_cap * head_dim * sizeof(float));
    float *new_sin = (float *)realloc(ctx->rope_cache_sin, (size_t)new_cap * head_dim * sizeof(float));
    if (!new_cos || !new_sin) return -1;
    ctx->rope_cache_cos = new_cos;
    ctx->rope_cache_sin = new_sin;

    /* Compute new positions only */
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float *cos_row = ctx->rope_cache_cos + (size_t)pos * head_dim;
        float *sin_row = ctx->rope_cache_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = (float)pos * ctx->rope_inv_freq[d];
            float c = cosf(angle), s = sinf(angle);
            cos_row[d] = c;
            cos_row[half + d] = c;
            sin_row[d] = s;
            sin_row[half + d] = s;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Buffer Allocation
 * ======================================================================== */

static void ensure_dec_buffers(tts_native_ctx_t *ctx) {
    if (ctx->dec_x) return;
    int dim = TTS_HIDDEN_SIZE;
    int q_dim = TTS_NUM_HEADS * TTS_HEAD_DIM;
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    int intermediate = TTS_INTERMEDIATE;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
}

static void ensure_prefill_buffers(tts_native_ctx_t *ctx, int seq_len) {
    if (ctx->pref_seq_cap >= seq_len) return;
    int dim = TTS_HIDDEN_SIZE;
    int q_dim = TTS_NUM_HEADS * TTS_HEAD_DIM;
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    int intermediate = TTS_INTERMEDIATE;

    free(ctx->pref_x);       free(ctx->pref_x_norm);
    free(ctx->pref_q);       free(ctx->pref_k);       free(ctx->pref_v);
    free(ctx->pref_attn_out); free(ctx->pref_proj_out);
    free(ctx->pref_ffn_out); free(ctx->pref_gate);     free(ctx->pref_gate_up);

    size_t s = (size_t)seq_len;
    ctx->pref_x        = (float *)malloc(s * dim * sizeof(float));
    ctx->pref_x_norm   = (float *)malloc(s * dim * sizeof(float));
    ctx->pref_q        = (float *)malloc(s * q_dim * sizeof(float));
    ctx->pref_k        = (float *)malloc(s * kv_dim * sizeof(float));
    ctx->pref_v        = (float *)malloc(s * kv_dim * sizeof(float));
    ctx->pref_attn_out = (float *)malloc(s * q_dim * sizeof(float));
    ctx->pref_proj_out = (float *)malloc(s * dim * sizeof(float));
    ctx->pref_ffn_out  = (float *)malloc(s * dim * sizeof(float));
    ctx->pref_gate     = (float *)malloc(s * 2 * intermediate * sizeof(float));
    ctx->pref_gate_up  = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->pref_seq_cap  = seq_len;
}

/* ========================================================================
 * Embedding Helpers
 * ======================================================================== */

/* Look up one token from bf16 embedding table, convert to f32. */
static void embed_lookup_bf16(float *dst, const uint16_t *table,
                              int token_id, int dim) {
    bf16_to_f32_row(dst, table + (size_t)token_id * dim, dim);
}

/* Embed N text tokens: table lookup (2048-dim) + MLP projection -> 1024-dim.
 * out: [N, 1024]. Allocates temporary buffers internally. */
static void text_embed_and_project(tts_native_ctx_t *ctx,
                                   const int64_t *ids, int n,
                                   float *out) {
    int text_dim = TTS_TEXT_HIDDEN_SIZE;  /* 2048 */
    int hidden = TTS_HIDDEN_SIZE;         /* 1024 */
    tts_text_project_t *tp = &ctx->text_project;
    int inter = tp->intermediate;

    /* Step 1: Look up text embeddings -> [N, 2048] */
    float *text_buf = (float *)malloc((size_t)n * text_dim * sizeof(float));
    for (int i = 0; i < n; i++) {
        embed_lookup_bf16(text_buf + i * text_dim,
                          ctx->text_embed_bf16, (int)ids[i], text_dim);
    }

    /* Step 2: fc1: [N, 2048] -> [N, intermediate] */
    float *inter_buf = (float *)malloc((size_t)n * inter * sizeof(float));
    linear_bf16(ctx, inter_buf, text_buf, tp->fc1_weight_bf16, n, text_dim, inter);
    if (tp->fc1_bias) {
        for (int i = 0; i < n; i++)
            qwen_add_inplace(inter_buf + i * inter, tp->fc1_bias, inter);
    }

    /* Step 3: SiLU activation in-place */
    qwen_silu(inter_buf, n * inter);

    /* Step 4: fc2: [N, intermediate] -> [N, 1024] */
    linear_bf16(ctx, out, inter_buf, tp->fc2_weight_bf16, n, inter, hidden);
    if (tp->fc2_bias) {
        for (int i = 0; i < n; i++)
            qwen_add_inplace(out + i * hidden, tp->fc2_bias, hidden);
    }

    free(text_buf);
    free(inter_buf);
}

/* Look up one codec token from the talker's codec embedding. */
static void codec_embed_lookup(tts_native_ctx_t *ctx, int token_id, float *out) {
    embed_lookup_bf16(out, ctx->codec_embed_bf16, token_id, TTS_HIDDEN_SIZE);
}

/* ========================================================================
 * Talker Prefill (multi-token forward pass)
 * ======================================================================== */

static void talker_prefill(tts_native_ctx_t *ctx, const float *input_embeds,
                           int seq_len) {
    qwen_decoder_t *dec = &ctx->talker;
    int dim = TTS_HIDDEN_SIZE;
    int n_heads = TTS_NUM_HEADS;
    int n_kv_heads = TTS_NUM_KV_HEADS;
    int head_dim = TTS_HEAD_DIM;
    int intermediate = TTS_INTERMEDIATE;
    float eps = TTS_RMS_NORM_EPS;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Initialize KV cache if needed */
    if (!ctx->kv_cache_k) {
        kv_cache_init(ctx, seq_len + 1024);
    } else if (seq_len > ctx->kv_cache_max) {
        kv_cache_grow(ctx, seq_len + 1024);
    }

    ensure_prefill_buffers(ctx, seq_len);
    if (ensure_rope_cache(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < TTS_TALKER_LAYERS; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(ctx->pref_x_norm, x, l->input_norm, seq_len, dim, eps);

        linear_bf16(ctx, ctx->pref_q, ctx->pref_x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        linear_bf16(ctx, ctx->pref_k, ctx->pref_x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        linear_bf16(ctx, ctx->pref_v, ctx->pref_x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);

        qwen_rms_norm_per_head(ctx->pref_q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(ctx->pref_k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        qwen_apply_rope_neox(ctx->pref_q, ctx->rope_cache_cos, ctx->rope_cache_sin,
                             seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(ctx->pref_k, ctx->rope_cache_cos, ctx->rope_cache_sin,
                             seq_len, n_kv_heads, head_dim);

        /* Store K/V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_k_at(ctx, layer, s), ctx->pref_k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_v_at(ctx, layer, s), ctx->pref_v + s * kv_dim, kv_dim * sizeof(float));
        }

        qwen_causal_attention(ctx->pref_attn_out, ctx->pref_q,
                              kv_k_at(ctx, layer, 0), kv_v_at(ctx, layer, 0),
                              seq_len, seq_len, n_heads, n_kv_heads, head_dim, scale, 0);

        linear_bf16(ctx, ctx->pref_proj_out, ctx->pref_attn_out, l->wo_weight_bf16,
                    seq_len, q_dim, dim);
        qwen_add_inplace(x, ctx->pref_proj_out, seq_len * dim);

        qwen_rms_norm(ctx->pref_x_norm, x, l->post_attn_norm, seq_len, dim, eps);

        linear_bf16(ctx, ctx->pref_gate, ctx->pref_x_norm, l->gate_up_fused_bf16,
                    seq_len, dim, 2 * intermediate);
        qwen_swiglu_multiply(ctx->pref_gate, ctx->pref_gate, seq_len, intermediate);
        linear_bf16(ctx, ctx->pref_ffn_out, ctx->pref_gate, l->down_weight_bf16,
                    seq_len, intermediate, dim);
        qwen_add_inplace(x, ctx->pref_ffn_out, seq_len * dim);
    }

    ctx->kv_cache_len = seq_len;

    /* Apply final norm to last position -> dec_x (used for first logits) */
    ensure_dec_buffers(ctx);
    float *last_hidden = x + (seq_len - 1) * dim;
    qwen_rms_norm(ctx->dec_x, last_hidden, dec->norm, 1, dim, eps);
}

/* ========================================================================
 * Talker Forward (single token -> fills logits_buf)
 * ======================================================================== */

/* Returns the un-normed hidden state in dec_x (for code predictor).
 * Fills logits_buf with [TTS_TALKER_VOCAB] logits. */
static void talker_forward(tts_native_ctx_t *ctx, const float *input_embed,
                           float *hidden_out) {
    qwen_decoder_t *dec = &ctx->talker;
    int dim = TTS_HIDDEN_SIZE;
    int n_heads = TTS_NUM_HEADS;
    int n_kv_heads = TTS_NUM_KV_HEADS;
    int head_dim = TTS_HEAD_DIM;
    int intermediate = TTS_INTERMEDIATE;
    float eps = TTS_RMS_NORM_EPS;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;
    if (pos >= ctx->kv_cache_max) {
        kv_cache_grow(ctx, pos + 1024);
    }
    if (ensure_rope_cache(ctx, pos + 1) != 0) return;

    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < TTS_TALKER_LAYERS; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(ctx->dec_x_norm, x, l->input_norm, 1, dim, eps);

        /* QKV projections */
        linear_bf16(ctx, ctx->dec_q, ctx->dec_x_norm, l->wq_weight_bf16, 1, dim, q_dim);
        linear_bf16(ctx, ctx->dec_k, ctx->dec_x_norm, l->wk_weight_bf16, 1, dim, kv_dim);
        linear_bf16(ctx, ctx->dec_v, ctx->dec_x_norm, l->wv_weight_bf16, 1, dim, kv_dim);

        qwen_rms_norm_per_head(ctx->dec_q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(ctx->dec_k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        qwen_apply_rope_neox(ctx->dec_q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(ctx->dec_k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_k_at(ctx, layer, pos), ctx->dec_k, kv_dim * sizeof(float));
        memcpy(kv_v_at(ctx, layer, pos), ctx->dec_v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        qwen_causal_attention(ctx->dec_attn_out, ctx->dec_q,
                              kv_k_at(ctx, layer, 0), kv_v_at(ctx, layer, 0),
                              1, total_seq, n_heads, n_kv_heads, head_dim, scale, pos);

        linear_bf16(ctx, ctx->dec_proj_out, ctx->dec_attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, ctx->dec_proj_out, dim);

        qwen_rms_norm(ctx->dec_x_norm, x, l->post_attn_norm, 1, dim, eps);

        linear_bf16(ctx, ctx->dec_gate, ctx->dec_x_norm, l->gate_up_fused_bf16,
                    1, dim, 2 * intermediate);
        qwen_swiglu_multiply(ctx->dec_gate, ctx->dec_gate, 1, intermediate);
        linear_bf16(ctx, ctx->dec_ffn_out, ctx->dec_gate, l->down_weight_bf16,
                    1, intermediate, dim);
        qwen_add_inplace(x, ctx->dec_ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + codec_head projection -> logits */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);

    /* Save NORMED hidden state for code predictor (matches HF last_hidden_state) */
    memcpy(hidden_out, x, dim * sizeof(float));

    linear_bf16(ctx, ctx->logits_buf, x, ctx->codec_head_bf16, 1, dim, TTS_TALKER_VOCAB);
}

/* ========================================================================
 * Code Predictor Forward (prefill-style, no persistent KV cache)
 *
 * Input: cp_embed_buf[seq_len, 1024] (projected_hidden + codec embeds)
 * Output: cp_logits_buf[TTS_CODEC_VOCAB] from lm_head[codebook_idx]
 * ======================================================================== */

static void code_predictor_forward(tts_native_ctx_t *ctx,
                                   int seq_len, int codebook_idx) {
    tts_code_predictor_t *cp = &ctx->code_pred;
    int dim = TTS_HIDDEN_SIZE;
    int n_heads = TTS_NUM_HEADS;
    int n_kv_heads = TTS_NUM_KV_HEADS;
    int head_dim = TTS_HEAD_DIM;
    int intermediate = TTS_INTERMEDIATE;
    float eps = TTS_RMS_NORM_EPS;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure prefill buffers are large enough for code predictor sequences */
    ensure_prefill_buffers(ctx, seq_len);
    if (ensure_rope_cache(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    memcpy(x, ctx->cp_embed_buf, (size_t)seq_len * dim * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < TTS_CODEPRED_LAYERS; layer++) {
        qwen_dec_layer_t *l = &cp->layers[layer];

        qwen_rms_norm(ctx->pref_x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV */
        linear_bf16(ctx, ctx->pref_q, ctx->pref_x_norm, l->wq_weight_bf16,
                    seq_len, dim, q_dim);
        linear_bf16(ctx, ctx->pref_k, ctx->pref_x_norm, l->wk_weight_bf16,
                    seq_len, dim, kv_dim);
        linear_bf16(ctx, ctx->pref_v, ctx->pref_x_norm, l->wv_weight_bf16,
                    seq_len, dim, kv_dim);

        qwen_rms_norm_per_head(ctx->pref_q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(ctx->pref_k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        qwen_apply_rope_neox(ctx->pref_q, ctx->rope_cache_cos, ctx->rope_cache_sin,
                             seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(ctx->pref_k, ctx->rope_cache_cos, ctx->rope_cache_sin,
                             seq_len, n_kv_heads, head_dim);

        /* Direct attention on Q/K/V (no KV cache needed) */
        qwen_causal_attention(ctx->pref_attn_out, ctx->pref_q, ctx->pref_k, ctx->pref_v,
                              seq_len, seq_len, n_heads, n_kv_heads, head_dim, scale, 0);

        linear_bf16(ctx, ctx->pref_proj_out, ctx->pref_attn_out, l->wo_weight_bf16,
                    seq_len, q_dim, dim);
        qwen_add_inplace(x, ctx->pref_proj_out, seq_len * dim);

        qwen_rms_norm(ctx->pref_x_norm, x, l->post_attn_norm, seq_len, dim, eps);

        linear_bf16(ctx, ctx->pref_gate, ctx->pref_x_norm, l->gate_up_fused_bf16,
                    seq_len, dim, 2 * intermediate);
        qwen_swiglu_multiply(ctx->pref_gate, ctx->pref_gate, seq_len, intermediate);
        linear_bf16(ctx, ctx->pref_ffn_out, ctx->pref_gate, l->down_weight_bf16,
                    seq_len, intermediate, dim);
        qwen_add_inplace(x, ctx->pref_ffn_out, seq_len * dim);
    }

    /* Final norm on last position */
    float *last_hidden = x + (seq_len - 1) * dim;
    float normed[TTS_HIDDEN_SIZE];
    qwen_rms_norm(normed, last_hidden, cp->norm, 1, dim, eps);

    /* Project through codebook-specific lm_head -> logits */
    linear_bf16(ctx, ctx->cp_logits_buf, normed,
                cp->lm_head_bf16[codebook_idx], 1, dim, TTS_CODEC_VOCAB);
}

/* ========================================================================
 * Input Token Sequence
 * ======================================================================== */

static int64_t *build_input_ids(tts_native_ctx_t *ctx, const char *text,
                                int *out_n_ids, int *out_role_len) {
    int n_role_cont = 0;
    int *role_cont = qwen_tokenizer_encode(ctx->tokenizer, "assistant\n", &n_role_cont);
    if (!role_cont || n_role_cont < 1) {
        fprintf(stderr, "TTS native: failed to encode role tokens\n");
        free(role_cont);
        return NULL;
    }

    int n_text = 0;
    int *text_tokens = qwen_tokenizer_encode(ctx->tokenizer, text, &n_text);
    if (!text_tokens || n_text < 1) {
        fprintf(stderr, "TTS native: failed to encode text\n");
        free(role_cont);
        free(text_tokens);
        return NULL;
    }

    int role_len = 1 + n_role_cont;
    int total = role_len + n_text + 5;
    int64_t *ids = (int64_t *)malloc((size_t)total * sizeof(int64_t));
    if (!ids) { free(role_cont); free(text_tokens); return NULL; }

    int pos = 0;
    ids[pos++] = TTS_TOKEN_IM_START;
    for (int i = 0; i < n_role_cont; i++)
        ids[pos++] = (int64_t)role_cont[i];
    for (int i = 0; i < n_text; i++)
        ids[pos++] = (int64_t)text_tokens[i];
    ids[pos++] = TTS_TOKEN_EOS;
    ids[pos++] = TTS_TOKEN_IM_END;
    ids[pos++] = TTS_TOKEN_IM_END;
    ids[pos++] = TTS_TOKEN_IM_END;
    ids[pos++] = TTS_TOKEN_IM_END;

    free(role_cont);
    free(text_tokens);
    *out_n_ids = total;
    *out_role_len = role_len;
    return ids;
}

/* ========================================================================
 * Prefill Embedding Assembly
 * ======================================================================== */

/* Build prefill embeddings and trailing text embeddings.
 * Matches the ONNX pipeline's build_prefill_embeddings exactly.
 * Returns 0 on success. Caller must free all output buffers. */
static int build_prefill_embeddings(tts_native_ctx_t *ctx,
                                    const int64_t *input_ids, int n_ids, int role_len,
                                    float **prefill_out, int *prefill_len_out,
                                    float **trailing_out, int *trailing_len_out,
                                    float **tts_pad_embed_out) {
    int H = TTS_HIDDEN_SIZE;

    /* 1. TTS special token embeddings via text_project */
    int64_t tts_special[] = { TTS_TOKEN_BOS, TTS_TOKEN_EOS, TTS_TOKEN_PAD };
    float *tts_special_embed = (float *)malloc(3 * H * sizeof(float));
    if (!tts_special_embed) return -1;
    text_embed_and_project(ctx, tts_special, 3, tts_special_embed);

    float *tts_bos_embed = tts_special_embed;
    float *tts_eos_embed = tts_special_embed + H;
    float *tts_pad_embed = tts_special_embed + 2 * H;

    /* Save pad embed for decode loop */
    *tts_pad_embed_out = (float *)malloc(H * sizeof(float));
    if (!*tts_pad_embed_out) { free(tts_special_embed); return -1; }
    memcpy(*tts_pad_embed_out, tts_pad_embed, H * sizeof(float));

    /* 2. Role embeddings via text_project */
    float *role_embed = (float *)malloc((size_t)role_len * H * sizeof(float));
    if (!role_embed) { free(tts_special_embed); free(*tts_pad_embed_out); return -1; }
    text_embed_and_project(ctx, input_ids, role_len, role_embed);

    /* 3. Codec prefix embeddings */
    int64_t codec_prefix[] = { TTS_CODEC_NOTHINK, TTS_CODEC_THINK_BOS,
                                TTS_CODEC_THINK_EOS, TTS_CODEC_PAD, TTS_CODEC_BOS };
    float codec_embed[5 * TTS_HIDDEN_SIZE];
    for (int i = 0; i < 5; i++)
        codec_embed_lookup(ctx, (int)codec_prefix[i], codec_embed + i * H);

    /* 4. First text token embedding via text_project */
    float text_first_embed[TTS_HIDDEN_SIZE];
    text_embed_and_project(ctx, &input_ids[role_len], 1, text_first_embed);

    /* 5. Assemble prefill: [role(role_len), pad+codec(4), text_first+codec(1)] */
    int pf_len = role_len + 5;
    float *pf = (float *)calloc((size_t)pf_len * H, sizeof(float));
    if (!pf) { free(tts_special_embed); free(role_embed); free(*tts_pad_embed_out); return -1; }

    memcpy(pf, role_embed, (size_t)role_len * H * sizeof(float));

    /* Sum: tts_pad + codec[0..2], tts_bos + codec[3] */
    int off = role_len * H;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < H; j++)
            pf[off + i * H + j] = tts_pad_embed[j] + codec_embed[i * H + j];
    }
    for (int j = 0; j < H; j++)
        pf[off + 3 * H + j] = tts_bos_embed[j] + codec_embed[3 * H + j];

    /* Sum: text_first + codec[4] */
    off = (role_len + 4) * H;
    for (int j = 0; j < H; j++)
        pf[off + j] = text_first_embed[j] + codec_embed[4 * H + j];

    *prefill_out = pf;
    *prefill_len_out = pf_len;

    /* 6. Trailing text embeddings */
    int text_remaining_start = role_len + 1;
    int text_remaining_end = n_ids - 5;
    int n_remaining = text_remaining_end - text_remaining_start;

    if (n_remaining > 0) {
        int trail_len = n_remaining + 1; /* +1 for tts_eos */
        float *trail = (float *)malloc((size_t)trail_len * H * sizeof(float));
        if (!trail) { free(tts_special_embed); free(role_embed); free(pf); free(*tts_pad_embed_out); return -1; }
        text_embed_and_project(ctx, &input_ids[text_remaining_start], n_remaining, trail);
        memcpy(trail + n_remaining * H, tts_eos_embed, H * sizeof(float));
        *trailing_out = trail;
        *trailing_len_out = trail_len;
    } else {
        float *trail = (float *)malloc(H * sizeof(float));
        if (!trail) { free(tts_special_embed); free(role_embed); free(pf); free(*tts_pad_embed_out); return -1; }
        memcpy(trail, tts_eos_embed, H * sizeof(float));
        *trailing_out = trail;
        *trailing_len_out = 1;
    }

    free(tts_special_embed);
    free(role_embed);
    return 0;
}

/* ========================================================================
 * Decode Loop
 * ======================================================================== */

static int64_t *run_decode_loop(tts_native_ctx_t *ctx,
                                float *prefill_embed, int prefill_seq_len,
                                float *trailing_embed, int trailing_seq_len,
                                float *tts_pad_embed,
                                float temperature, int top_k,
                                int *out_n_steps) {
    int H = TTS_HIDDEN_SIZE;
    *out_n_steps = 0;

    int64_t *all_codes = (int64_t *)malloc((size_t)TTS_MAX_DECODE_STEPS * TTS_NUM_CODE_GROUPS * sizeof(int64_t));
    int *cb0_history = (int *)malloc(TTS_MAX_DECODE_STEPS * sizeof(int));
    if (!all_codes || !cb0_history) { free(all_codes); free(cb0_history); return NULL; }
    int history_len = 0;

    /* Allocate code predictor input buffer */
    if (!ctx->cp_embed_buf) {
        ctx->cp_embed_buf = (float *)malloc(18 * H * sizeof(float));
        if (!ctx->cp_embed_buf) { free(all_codes); free(cb0_history); return NULL; }
    }

    /* Allocate logits buffers */
    if (!ctx->logits_buf) {
        ctx->logits_buf = (float *)malloc(TTS_TALKER_VOCAB * sizeof(float));
        if (!ctx->logits_buf) { free(all_codes); free(cb0_history); return NULL; }
    }
    if (!ctx->cp_logits_buf) {
        ctx->cp_logits_buf = (float *)malloc(TTS_CODEC_VOCAB * sizeof(float));
        if (!ctx->cp_logits_buf) { free(all_codes); free(cb0_history); return NULL; }
    }

    /* ---- Prefill ---- */
    if (ctx->verbose) printf("  TTS native prefill: seq_len=%d\n", prefill_seq_len);
    talker_prefill(ctx, prefill_embed, prefill_seq_len);

    /* After prefill, dec_x contains final-normed hidden state of last position.
     * Compute initial logits from it. */
    linear_bf16(ctx, ctx->logits_buf, ctx->dec_x, ctx->codec_head_bf16,
                1, TTS_HIDDEN_SIZE, TTS_TALKER_VOCAB);

    /* Code predictor needs the NORMED hidden state (same as what lm_head sees).
     * In HuggingFace, model.last_hidden_state is after final RMSNorm.
     * dec_x already contains the normed state from talker_prefill. */
    float *last_hidden_normed = (float *)malloc(H * sizeof(float));
    if (!last_hidden_normed) { free(all_codes); free(cb0_history); return NULL; }
    memcpy(last_hidden_normed, ctx->dec_x, H * sizeof(float));

    int suppress_start = TTS_TALKER_VOCAB - 1024; /* 2048 */
    int n_steps = 0;

    extern volatile int g_shutdown;

    for (int step = 0; step < TTS_MAX_DECODE_STEPS; step++) {
        if (g_shutdown) {
            if (ctx->verbose) printf("  TTS native: interrupted by shutdown\n");
            break;
        }

        /* ---- Sample cb0 from talker logits ---- */
        float *logits = ctx->logits_buf;

        /* Debug: print logit diagnostics for first few steps */
        if (ctx->verbose && step < 5) {
            float max_logit = logits[0];
            int max_idx = 0;
            for (int i = 1; i < suppress_start; i++) {
                if (logits[i] > max_logit) { max_logit = logits[i]; max_idx = i; }
            }
            printf("  [step %d] max_logit=%.2f (idx=%d), eos_logit=%.2f (idx=%d)\n",
                   step, max_logit, max_idx, logits[TTS_CODEC_EOS], TTS_CODEC_EOS);
        }

        /* Suppress [2048, 3072) except EOS */
        for (int i = suppress_start; i < TTS_TALKER_VOCAB; i++) {
            if (i != TTS_CODEC_EOS) logits[i] = -1.0e9f;
        }

        tts_apply_repetition_penalty(logits, TTS_TALKER_VOCAB,
                                     cb0_history, history_len, 1.05f);

        int cb0 = tts_sample_topk(logits, TTS_TALKER_VOCAB, temperature, top_k);

        if (cb0 == TTS_CODEC_EOS) {
            if (ctx->verbose) printf("  TTS native: EOS at step %d\n", step);
            break;
        }

        cb0_history[history_len++] = cb0;

        /* Store cb0 */
        int64_t *step_codes = &all_codes[n_steps * TTS_NUM_CODE_GROUPS];
        step_codes[0] = (int64_t)cb0;

        /* ---- Code predictor: sub-codes 1-15 ---- */

        /* Build initial code predictor input: [projected_hidden, cb0_embed] */
        tts_code_predictor_t *cp = &ctx->code_pred;

        /* Project talker hidden through mtp_projection (or pass through) */
        if (cp->mtp_proj_weight) {
            float projected[TTS_HIDDEN_SIZE];
            qwen_linear_nobias(projected, last_hidden_normed, cp->mtp_proj_weight,
                               1, H, H);
            if (cp->mtp_proj_bias)
                qwen_add_inplace(projected, cp->mtp_proj_bias, H);
            memcpy(ctx->cp_embed_buf, projected, H * sizeof(float));
        } else {
            /* Same hidden size -- pass through directly */
            memcpy(ctx->cp_embed_buf, last_hidden_normed, H * sizeof(float));
        }

        /* cb0 embedding (from talker's codec embedding) */
        codec_embed_lookup(ctx, cb0, ctx->cp_embed_buf + H);

        /* codec_sum accumulates all 16 codec embeddings */
        float codec_sum[TTS_HIDDEN_SIZE];
        memcpy(codec_sum, ctx->cp_embed_buf + H, H * sizeof(float)); /* start with cb0 embed */

        int sub_seq_len = 2;

        for (int j = 0; j < TTS_NUM_CODE_GROUPS - 1; j++) {
            /* Run code predictor */
            code_predictor_forward(ctx, sub_seq_len, j);

            /* Sample sub-code */
            int sub_tok = tts_sample_topk(ctx->cp_logits_buf, TTS_CODEC_VOCAB,
                                          temperature, top_k);
            step_codes[j + 1] = (int64_t)sub_tok;

            /* Get sub-code embedding from code predictor's per-codebook table */
            float sub_embed[TTS_HIDDEN_SIZE];
            embed_lookup_bf16(sub_embed, cp->codec_embed_bf16[j],
                              sub_tok, TTS_HIDDEN_SIZE);

            /* Append to code predictor input for next iteration */
            memcpy(ctx->cp_embed_buf + sub_seq_len * H, sub_embed, H * sizeof(float));
            sub_seq_len++;

            /* Accumulate into codec_sum */
            qwen_add_inplace(codec_sum, sub_embed, H);
        }

        /* ---- Build next decode step input ---- */
        /* codec_sum += trailing text embed (or pad) */
        if (step < trailing_seq_len) {
            qwen_add_inplace(codec_sum, trailing_embed + step * H, H);
        } else {
            qwen_add_inplace(codec_sum, tts_pad_embed, H);
        }

        n_steps++;

        if (step + 1 >= TTS_MAX_DECODE_STEPS) break;

        /* ---- Talker decode step ---- */
        talker_forward(ctx, codec_sum, last_hidden_normed);
        /* logits_buf and last_hidden_normed are now updated for next iteration */

        if (ctx->verbose && (step % 50 == 0 || step < 5)) {
            printf("  TTS native step %d: cb0=%d\n", step, cb0);
        }
    }

    free(last_hidden_normed);
    free(cb0_history);

    *out_n_steps = n_steps;
    return all_codes;
}

/* ========================================================================
 * GPU Weight Upload
 * ======================================================================== */

#ifdef USE_CUBLAS
static int upload_gpu_weights(tts_native_ctx_t *ctx) {
    qwen_gpu_ctx_t *gpu = (qwen_gpu_ctx_t *)ctx->gpu_ctx;
    if (!gpu) return 0;

    int dim = TTS_HIDDEN_SIZE;
    int q_dim = TTS_NUM_HEADS * TTS_HEAD_DIM;
    int kv_dim = TTS_NUM_KV_HEADS * TTS_HEAD_DIM;
    int inter = TTS_INTERMEDIATE;

    if (ctx->verbose) printf("TTS native: uploading talker weights to GPU...\n");

    for (int i = 0; i < TTS_TALKER_LAYERS; i++) {
        qwen_dec_layer_t *l = &ctx->talker.layers[i];
        qwen_gpu_upload_weight_bf16(gpu, l->wq_weight_bf16, q_dim, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->wk_weight_bf16, kv_dim, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->wv_weight_bf16, kv_dim, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->wo_weight_bf16, dim, q_dim);
        qwen_gpu_upload_weight_bf16(gpu, l->gate_up_fused_bf16, 2 * inter, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->down_weight_bf16, dim, inter);
    }

    /* Codec head */
    qwen_gpu_upload_weight_bf16(gpu, ctx->codec_head_bf16, TTS_TALKER_VOCAB, dim);

    /* Text projection */
    tts_text_project_t *tp = &ctx->text_project;
    qwen_gpu_upload_weight_bf16(gpu, tp->fc1_weight_bf16, tp->intermediate, TTS_TEXT_HIDDEN_SIZE);
    qwen_gpu_upload_weight_bf16(gpu, tp->fc2_weight_bf16, dim, tp->intermediate);

    /* Code predictor layers (5 layers) */
    if (ctx->verbose) printf("TTS native: uploading code predictor weights to GPU...\n");

    for (int i = 0; i < TTS_CODEPRED_LAYERS; i++) {
        qwen_dec_layer_t *l = &ctx->code_pred.layers[i];
        qwen_gpu_upload_weight_bf16(gpu, l->wq_weight_bf16, q_dim, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->wk_weight_bf16, kv_dim, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->wv_weight_bf16, kv_dim, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->wo_weight_bf16, dim, q_dim);
        qwen_gpu_upload_weight_bf16(gpu, l->gate_up_fused_bf16, 2 * inter, dim);
        qwen_gpu_upload_weight_bf16(gpu, l->down_weight_bf16, dim, inter);
    }

    /* Code predictor lm_heads (15 codebook-specific heads) */
    for (int j = 0; j < TTS_NUM_CODE_GROUPS - 1; j++) {
        qwen_gpu_upload_weight_bf16(gpu, ctx->code_pred.lm_head_bf16[j],
                                    TTS_CODEC_VOCAB, dim);
    }

    if (ctx->verbose) {
        qwen_gpu_print_stats(gpu);
    }

    return 0;
}
#endif

/* ========================================================================
 * Public API
 * ======================================================================== */

int tts_native_init(tts_native_ctx_t *ctx, const char *model_dir,
                    void *gpu_ctx, int verbose) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->verbose = verbose;

#ifdef USE_CUBLAS
    ctx->gpu_ctx = gpu_ctx;
#else
    (void)gpu_ctx;
#endif

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    if (verbose) printf("TTS native: loading model from %s\n", model_dir);

    /* Open safetensors file(s) */
    ctx->safetensors = multi_safetensors_open(model_dir);
    if (!ctx->safetensors) {
        fprintf(stderr, "TTS native: failed to open safetensors in %s\n", model_dir);
        return -1;
    }

    /* Set up talker config (identical to qwen-asr 0.6B decoder) */
    ctx->talker_config.dec_hidden = TTS_HIDDEN_SIZE;
    ctx->talker_config.dec_layers = TTS_TALKER_LAYERS;
    ctx->talker_config.dec_heads = TTS_NUM_HEADS;
    ctx->talker_config.dec_kv_heads = TTS_NUM_KV_HEADS;
    ctx->talker_config.dec_head_dim = TTS_HEAD_DIM;
    ctx->talker_config.dec_intermediate = TTS_INTERMEDIATE;
    ctx->talker_config.vocab_size = TTS_TALKER_VOCAB;
    ctx->talker_config.dec_rms_norm_eps = TTS_RMS_NORM_EPS;
    ctx->talker_config.dec_rope_theta = TTS_ROPE_THETA;

    /* Load weights */
    if (load_talker_weights(ctx) != 0) goto fail;
    if (verbose) printf("TTS native: talker weights loaded (28 layers)\n");

    if (load_code_predictor_weights(ctx) != 0) goto fail;
    if (verbose) printf("TTS native: code predictor weights loaded (5 layers)\n");

    if (load_embedding_weights(ctx) != 0) goto fail;
    if (verbose) printf("TTS native: embedding weights loaded\n");

    /* Load tokenizer */
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    ctx->tokenizer = qwen_tokenizer_load(vocab_path);
    if (!ctx->tokenizer) {
        fprintf(stderr, "TTS native: failed to load tokenizer from %s\n", vocab_path);
        goto fail;
    }
    if (verbose) printf("TTS native: tokenizer loaded (%d tokens)\n",
                        ctx->tokenizer->vocab_size);

#ifdef USE_CUBLAS
    if (ctx->gpu_ctx) {
        const char *cpu_only = getenv("TTS_CPU_ONLY");
        if (cpu_only && cpu_only[0] == '1') {
            if (verbose) printf("TTS native: TTS_CPU_ONLY=1, skipping GPU upload\n");
            ctx->gpu_ctx = NULL;
        } else {
            upload_gpu_weights(ctx);
        }
    }
#endif

    QueryPerformanceCounter(&t_end);
    double load_ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
    if (verbose) printf("TTS native: model loaded in %.0f ms\n", load_ms);

    return 0;

fail:
    tts_native_free(ctx);
    return -1;
}

void tts_native_free(tts_native_ctx_t *ctx) {
    /* Free fused weights (allocated, not mmap'd) */
    for (int i = 0; i < TTS_TALKER_LAYERS; i++)
        free(ctx->talker.layers[i].gate_up_fused_bf16);
    for (int i = 0; i < TTS_CODEPRED_LAYERS; i++)
        free(ctx->code_pred.layers[i].gate_up_fused_bf16);

    /* Free f32 weights (allocated by safetensors_get_f32) */
    for (int i = 0; i < TTS_TALKER_LAYERS; i++) {
        free(ctx->talker.layers[i].q_norm_weight);
        free(ctx->talker.layers[i].k_norm_weight);
        free(ctx->talker.layers[i].input_norm);
        free(ctx->talker.layers[i].post_attn_norm);
    }
    free(ctx->talker.norm);

    for (int i = 0; i < TTS_CODEPRED_LAYERS; i++) {
        free(ctx->code_pred.layers[i].q_norm_weight);
        free(ctx->code_pred.layers[i].k_norm_weight);
        free(ctx->code_pred.layers[i].input_norm);
        free(ctx->code_pred.layers[i].post_attn_norm);
    }
    free(ctx->code_pred.norm);
    free(ctx->code_pred.mtp_proj_weight);
    free(ctx->code_pred.mtp_proj_bias);

    free(ctx->text_project.fc1_bias);
    free(ctx->text_project.fc2_bias);

    /* Free KV cache */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);

    /* Free decode buffers */
    free(ctx->dec_x);       free(ctx->dec_x_norm);
    free(ctx->dec_q);       free(ctx->dec_k);       free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out);
    free(ctx->dec_gate);    free(ctx->dec_ffn_out);

    /* Free prefill buffers */
    free(ctx->pref_x);       free(ctx->pref_x_norm);
    free(ctx->pref_q);       free(ctx->pref_k);       free(ctx->pref_v);
    free(ctx->pref_attn_out); free(ctx->pref_proj_out);
    free(ctx->pref_ffn_out); free(ctx->pref_gate);

    /* Free misc buffers */
    free(ctx->cp_embed_buf);
    free(ctx->rope_cache_cos);
    free(ctx->rope_cache_sin);
    free(ctx->rope_inv_freq);
    free(ctx->logits_buf);
    free(ctx->cp_logits_buf);
    free(ctx->tmp_embed);
    free(ctx->tmp_text_embed);

    /* Free tokenizer */
    if (ctx->tokenizer)
        qwen_tokenizer_free(ctx->tokenizer);

    /* Close safetensors (unmaps files) */
    if (ctx->safetensors)
        multi_safetensors_close(ctx->safetensors);

    memset(ctx, 0, sizeof(*ctx));
}

int tts_native_decode(tts_native_ctx_t *ctx, const char *text,
                      float temperature, int top_k,
                      int64_t **codes_out, int *n_steps_out) {
    *codes_out = NULL;
    *n_steps_out = 0;

    LARGE_INTEGER freq, t_start;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    if (ctx->verbose) printf("TTS native decode: \"%s\"\n", text);

    /* Reset KV cache for new sequence */
    ctx->kv_cache_len = 0;

    /* 1. Tokenize */
    int n_ids = 0, role_len = 0;
    int64_t *input_ids = build_input_ids(ctx, text, &n_ids, &role_len);
    if (!input_ids) return -1;

    if (ctx->verbose)
        printf("  tokens: %d total, %d role, %d text, 5 end\n",
               n_ids, role_len, n_ids - role_len - 5);

    /* 2. Build embeddings */
    float *prefill_data = NULL, *trailing_data = NULL, *tts_pad_embed = NULL;
    int prefill_len = 0, trailing_len = 0;

    int rc = build_prefill_embeddings(ctx, input_ids, n_ids, role_len,
                                       &prefill_data, &prefill_len,
                                       &trailing_data, &trailing_len,
                                       &tts_pad_embed);
    free(input_ids);
    if (rc != 0) return -1;

    if (ctx->verbose)
        printf("  prefill seq_len=%d, trailing=%d\n", prefill_len, trailing_len);

    /* 3. Decode loop */
    int n_steps = 0;
    int64_t *codes = run_decode_loop(ctx, prefill_data, prefill_len,
                                      trailing_data, trailing_len,
                                      tts_pad_embed, temperature, top_k,
                                      &n_steps);
    free(prefill_data);
    free(trailing_data);
    free(tts_pad_embed);

    if (!codes || n_steps == 0) {
        free(codes);
        fprintf(stderr, "TTS native: decode produced no steps\n");
        return -1;
    }

    LARGE_INTEGER t_end;
    QueryPerformanceCounter(&t_end);
    double elapsed_ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;

    if (ctx->verbose)
        printf("  TTS native: %d steps (%.1fs audio) in %.0f ms\n",
               n_steps, n_steps * 0.08, elapsed_ms);

    *codes_out = codes;
    *n_steps_out = n_steps;
    return 0;
}
