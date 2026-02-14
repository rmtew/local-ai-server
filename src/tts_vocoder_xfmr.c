/*
 * tts_vocoder_xfmr.c - Pre-transformer for Qwen3-TTS vocoder
 *
 * 8-layer Qwen2-style transformer.
 * hidden=512 (residual), attn_dim=1024 (Q/K/V), heads=16, head_dim=64.
 * No GQA, no per-head Q/K norm, LayerScale after attention and MLP.
 * RoPE theta=10000. Weights are f32.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_vocoder.h"
#include "qwen_asr_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

/* ========================================================================
 * Helpers
 * ======================================================================== */

/* f32 linear: out[M,N] = in[M,K] @ W[N,K]^T */
static void linear_f32(float *out, const float *in, const float *W,
                       int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, in, K, W, K,
                0.0f, out, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += in[m * K + k] * W[n * K + k];
            }
            out[m * N + n] = sum;
        }
    }
#endif
}

/* f32 linear with bias: out[M,N] = in[M,K] @ W[N,K]^T + bias[N] */
static void linear_f32_bias(float *out, const float *in, const float *W,
                             const float *bias, int M, int K, int N) {
    linear_f32(out, in, W, M, K, N);
    if (bias) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                out[m * N + n] += bias[n];
            }
        }
    }
}

/* LayerScale: x[t, h] *= gamma[h] */
static void layer_scale(float *x, const float *gamma, int T, int hidden) {
    for (int t = 0; t < T; t++) {
        float *row = x + t * hidden;
        for (int h = 0; h < hidden; h++) {
            row[h] *= gamma[h];
        }
    }
}

/* SwiGLU from separate gate and up buffers.
 * gate: [T, inter], up: [T, inter] -> out: [T, inter]
 * out[i] = SiLU(gate[i]) * up[i] */
static void swiglu_separate(float *out, const float *gate, const float *up,
                             int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = (g * sig) * up[i];
    }
}

/* ========================================================================
 * Ensure transformer scratch buffers
 * ======================================================================== */

static void ensure_xfmr_buffers(tts_vocoder_ctx_t *ctx, int T) {
    if (T <= ctx->xfmr_buf_cap) return;

    free(ctx->xfmr_q);
    free(ctx->xfmr_k);
    free(ctx->xfmr_v);
    free(ctx->xfmr_attn_out);
    free(ctx->xfmr_proj_out);
    free(ctx->xfmr_norm_buf);
    free(ctx->xfmr_gate_up);
    free(ctx->xfmr_ffn_out);
    free(ctx->rope_cos);
    free(ctx->rope_sin);

    int h = VOC_XFMR_HIDDEN;       /* 512 */
    int ad = VOC_XFMR_ATTN_DIM;    /* 1024 */
    int inter = VOC_XFMR_INTERMEDIATE; /* 1024 */
    int hd = VOC_XFMR_HEAD_DIM;    /* 64 */

    ctx->xfmr_q        = (float *)malloc((size_t)T * ad * sizeof(float));
    ctx->xfmr_k        = (float *)malloc((size_t)T * ad * sizeof(float));
    ctx->xfmr_v        = (float *)malloc((size_t)T * ad * sizeof(float));
    ctx->xfmr_attn_out = (float *)malloc((size_t)T * ad * sizeof(float));
    ctx->xfmr_proj_out = (float *)malloc((size_t)T * h * sizeof(float));
    ctx->xfmr_norm_buf = (float *)malloc((size_t)T * h * sizeof(float));
    /* gate and up stored consecutively */
    ctx->xfmr_gate_up  = (float *)malloc((size_t)T * inter * 2 * sizeof(float));
    ctx->xfmr_ffn_out  = (float *)malloc((size_t)T * inter * sizeof(float));
    ctx->rope_cos      = (float *)malloc((size_t)T * hd * sizeof(float));
    ctx->rope_sin      = (float *)malloc((size_t)T * hd * sizeof(float));

    ctx->xfmr_buf_cap = T;

    (void)ad;
}

/* ========================================================================
 * Pre-transformer forward pass
 * ======================================================================== */

void voc_pre_transformer(tts_vocoder_ctx_t *ctx, float *out, const float *in, int T) {
    voc_pre_transformer_t *xf = &ctx->xfmr;
    int h = VOC_XFMR_HIDDEN;       /* 512 */
    int ad = VOC_XFMR_ATTN_DIM;    /* 1024 */
    int inter = VOC_XFMR_INTERMEDIATE; /* 1024 */
    int heads = VOC_XFMR_HEADS;    /* 16 */
    int hd = VOC_XFMR_HEAD_DIM;    /* 64 */
    float scale = 1.0f / sqrtf((float)hd);

    ensure_xfmr_buffers(ctx, T);

    /* Input is [1024, T] (channels-first). Transpose to [T, 1024] */
    float *x_t1024 = ctx->buf_a; /* reuse ping-pong buf for transpose */
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < 1024; c++) {
            x_t1024[t * 1024 + c] = in[(size_t)c * T + t];
        }
    }

    /* input_proj: [T, 1024] -> [T, 512] with optional bias */
    float *x_h = (float *)malloc((size_t)T * h * sizeof(float));
    linear_f32_bias(x_h, x_t1024, xf->input_proj, xf->input_proj_bias, T, 1024, h);

    /* RoPE cos/sin for positions 0..T-1 */
    {
        int *positions = (int *)malloc((size_t)T * sizeof(int));
        for (int t = 0; t < T; t++) positions[t] = t;
        qwen_compute_rope_neox(ctx->rope_cos, ctx->rope_sin, positions,
                               T, hd, VOC_XFMR_ROPE_THETA);
        free(positions);
    }

    /* 8-layer transformer */
    for (int layer = 0; layer < VOC_XFMR_LAYERS; layer++) {
        voc_xfmr_layer_t *l = &xf->layers[layer];

        /* RMSNorm on hidden */
        qwen_rms_norm(ctx->xfmr_norm_buf, x_h, l->input_norm,
                       T, h, VOC_XFMR_RMS_EPS);

        /* Q, K, V: [T, 512] -> [T, 1024] each */
        linear_f32(ctx->xfmr_q, ctx->xfmr_norm_buf, l->wq, T, h, ad);
        linear_f32(ctx->xfmr_k, ctx->xfmr_norm_buf, l->wk, T, h, ad);
        linear_f32(ctx->xfmr_v, ctx->xfmr_norm_buf, l->wv, T, h, ad);

        /* Apply RoPE */
        qwen_apply_rope_neox(ctx->xfmr_q, ctx->rope_cos, ctx->rope_sin,
                              T, heads, hd);
        qwen_apply_rope_neox(ctx->xfmr_k, ctx->rope_cos, ctx->rope_sin,
                              T, heads, hd);

        /* Causal attention (no GQA: n_kv_heads == n_heads == 16) */
        qwen_causal_attention(ctx->xfmr_attn_out, ctx->xfmr_q, ctx->xfmr_k,
                               ctx->xfmr_v, T, T, heads, heads, hd, scale, 0);

        /* O projection: [T, 1024] -> [T, 512] */
        linear_f32(ctx->xfmr_proj_out, ctx->xfmr_attn_out, l->wo, T, ad, h);

        /* LayerScale + residual */
        layer_scale(ctx->xfmr_proj_out, l->attn_layer_scale, T, h);
        qwen_add_inplace(x_h, ctx->xfmr_proj_out, T * h);

        /* Post-attention RMSNorm */
        qwen_rms_norm(ctx->xfmr_norm_buf, x_h, l->post_attn_norm,
                       T, h, VOC_XFMR_RMS_EPS);

        /* SwiGLU MLP: gate [T, 1024], up [T, 1024] computed separately */
        float *gate_buf = ctx->xfmr_gate_up;
        float *up_buf = ctx->xfmr_gate_up + (size_t)T * inter;
        linear_f32(gate_buf, ctx->xfmr_norm_buf, l->gate_weight, T, h, inter);
        linear_f32(up_buf, ctx->xfmr_norm_buf, l->up_weight, T, h, inter);

        /* SiLU(gate) * up -> ffn_out [T, 1024] */
        swiglu_separate(ctx->xfmr_ffn_out, gate_buf, up_buf, T * inter);

        /* Down projection: [T, 1024] -> [T, 512] */
        linear_f32(ctx->xfmr_proj_out, ctx->xfmr_ffn_out, l->down_weight, T, inter, h);

        /* LayerScale + residual */
        layer_scale(ctx->xfmr_proj_out, l->mlp_layer_scale, T, h);
        qwen_add_inplace(x_h, ctx->xfmr_proj_out, T * h);

    }

    /* Final RMSNorm */
    qwen_rms_norm(x_h, x_h, xf->final_norm, T, h, VOC_XFMR_RMS_EPS);

    /* output_proj: [T, 512] -> [T, 1024] with optional bias.
     * Write to buf_b (not x_t1024/buf_a) because out == buf_a and the
     * subsequent transpose would self-overwrite if source == dest. */
    float *x_out = ctx->buf_b; /* in (buf_b) was consumed by initial transpose */
    linear_f32_bias(x_out, x_h, xf->output_proj, xf->output_proj_bias, T, h, 1024);
    free(x_h);

    /* Transpose [T, 1024] back to [1024, T] */
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < 1024; c++) {
            out[(size_t)c * T + t] = x_out[t * 1024 + c];
        }
    }
}
