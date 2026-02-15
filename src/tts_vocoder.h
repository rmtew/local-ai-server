/*
 * tts_vocoder.h - Native C vocoder for Qwen3-TTS Tokenizer-12Hz
 *
 * Replaces the ONNX Runtime vocoder (tokenizer12hz_decode.onnx) with a native
 * C implementation using im2col+SGEMM (OpenBLAS) for Conv1d operations.
 *
 * Architecture: RVQ decode -> CausalConv1d -> 8-layer transformer ->
 *               ConvNeXt upsample (4x) -> BigVGAN decoder (480x) -> audio
 * Total upsample: 1920x (12.5 Hz codec frames -> 24 kHz audio)
 */

#ifndef TTS_VOCODER_H
#define TTS_VOCODER_H

#include "qwen_asr_safetensors.h"
#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define VOC_NUM_CODEBOOKS       16
#define VOC_CODEBOOK_SIZE       2048  /* entries per codebook (all 16) */
#define VOC_CODEBOOK_DIM        256
#define VOC_RVQ_OUT_DIM         512   /* after output_proj sum */

/* Pre-conv */
#define VOC_PRE_CONV_IN         512
#define VOC_PRE_CONV_OUT        1024
#define VOC_PRE_CONV_KERNEL     3

/* Pre-transformer (8-layer Qwen2)
 * hidden=512 (residual stream), attn_dim=1024 (Q/K/V projection dim)
 * heads=16, head_dim=64, so attn_dim = 16*64 = 1024 */
#define VOC_XFMR_LAYERS        8
#define VOC_XFMR_HIDDEN        512
#define VOC_XFMR_ATTN_DIM      1024  /* heads * head_dim */
#define VOC_XFMR_HEADS         16
#define VOC_XFMR_HEAD_DIM      64
#define VOC_XFMR_INTERMEDIATE  1024  /* SwiGLU MLP intermediate */
#define VOC_XFMR_ROPE_THETA    10000.0f
#define VOC_XFMR_RMS_EPS       1e-5f

/* ConvNeXt upsample */
#define VOC_UPSAMPLE_STAGES    2
#define VOC_UPSAMPLE_CHANNELS  1024
#define VOC_CONVNEXT_KERNEL    7     /* depthwise conv */

/* BigVGAN decoder */
#define VOC_BIGVGAN_INIT_CH    1536
#define VOC_BIGVGAN_NUM_BLOCKS 4
/* Upsample rates per block: 8, 5, 4, 3 */
/* Channel progression: 1536 -> 768 -> 384 -> 192 -> 96 */
/* ResUnit dilations: 1, 3, 9 */
#define VOC_BIGVGAN_RESUNITS   3
#define VOC_BIGVGAN_RES_KERNEL 7

#define VOC_SAMPLE_RATE        24000
#define VOC_UPSAMPLE_TOTAL     1920  /* 2*2 * 8*5*4*3 */

/* ========================================================================
 * Weight Structures
 * ======================================================================== */

/* RVQ codebook (one of 16) */
typedef struct {
    float *embeddings;      /* [2048, 256] normalized at load */
    float *proj_weight;     /* [512, 256, 1] = [512, 256] conv1d 1x1 */
} voc_codebook_t;

/* Transformer layer (Qwen2-style, no GQA, no per-head Q/K norm) */
typedef struct {
    float *input_norm;      /* [512] RMSNorm */
    float *wq;              /* [1024, 512] f32 -- attn_dim x hidden */
    float *wk;              /* [1024, 512] f32 */
    float *wv;              /* [1024, 512] f32 */
    float *wo;              /* [512, 1024] f32 -- hidden x attn_dim */
    float *post_attn_norm;  /* [512] RMSNorm */
    float *gate_weight;     /* [1024, 512] f32 */
    float *up_weight;       /* [1024, 512] f32 */
    float *down_weight;     /* [512, 1024] f32 */
    float *attn_layer_scale;  /* [512] elementwise */
    float *mlp_layer_scale;   /* [512] elementwise */
} voc_xfmr_layer_t;

/* Pre-transformer block */
typedef struct {
    float *input_proj;      /* [512, 1024] */
    float *input_proj_bias; /* [512] */
    float *output_proj;     /* [1024, 512] */
    float *output_proj_bias;/* [1024] */
    float *final_norm;      /* [512] RMSNorm */
    voc_xfmr_layer_t layers[VOC_XFMR_LAYERS];
} voc_pre_transformer_t;

/* ConvNeXt block (within upsample stage) */
typedef struct {
    float *dw_weight;       /* [channels, 1, 7] depthwise */
    float *dw_bias;         /* [channels] */
    float *norm_weight;     /* [channels] LayerNorm */
    float *norm_bias;       /* [channels] */
    float *pw1_weight;      /* [channels*4, channels] pointwise up */
    float *pw1_bias;        /* [channels*4] */
    float *pw2_weight;      /* [channels, channels*4] pointwise down */
    float *pw2_bias;        /* [channels] */
    float *gamma;           /* [channels] layer scale */
} voc_convnext_t;

/* Upsample stage: transposed conv + ConvNeXt */
typedef struct {
    float *tconv_weight;    /* [in_ch, out_ch, kernel] */
    float *tconv_bias;      /* [out_ch] */
    int stride;             /* = kernel size for this stage (2) */
    voc_convnext_t convnext;
} voc_upsample_stage_t;

/* BigVGAN residual unit: SnakeBeta -> Conv(k=7,dil=d) -> SnakeBeta -> Conv(k=1) */
typedef struct {
    float *snake1_alpha;    /* [channels] */
    float *snake1_beta;     /* [channels] */
    float *conv1_weight;    /* [channels, channels, 7] */
    float *conv1_bias;      /* [channels] */
    float *snake2_alpha;    /* [channels] */
    float *snake2_beta;     /* [channels] */
    float *conv2_weight;    /* [channels, channels, 1] */
    float *conv2_bias;      /* [channels] */
    int dilation;
} voc_resunit_t;

/* BigVGAN upsample block: SnakeBeta -> TransConv -> 3x ResUnit */
typedef struct {
    float *snake_alpha;     /* [in_ch] */
    float *snake_beta;      /* [in_ch] */
    float *tconv_weight;    /* [in_ch, out_ch, 2*rate] */
    float *tconv_bias;      /* [out_ch] */
    int rate;               /* upsample rate (8,5,4,3) */
    int in_ch;
    int out_ch;
    voc_resunit_t resunits[VOC_BIGVGAN_RESUNITS]; /* dilation 1,3,9 */
} voc_bigvgan_block_t;

/* BigVGAN decoder */
typedef struct {
    float *init_weight;     /* [1536, 1024, 7] */
    float *init_bias;       /* [1536] */
    voc_bigvgan_block_t blocks[VOC_BIGVGAN_NUM_BLOCKS];
    float *final_snake_alpha; /* [96] */
    float *final_snake_beta;  /* [96] */
    float *final_weight;    /* [1, 96, 7] */
    float *final_bias;      /* [1] */
} voc_bigvgan_t;

/* ========================================================================
 * Vocoder Context
 * ======================================================================== */

typedef struct {
    /* Weights */
    voc_codebook_t codebooks[VOC_NUM_CODEBOOKS];
    float *pre_conv_weight;     /* [1024, 512, 3] */
    float *pre_conv_bias;       /* [1024] */
    voc_pre_transformer_t xfmr;
    voc_upsample_stage_t upsample[VOC_UPSAMPLE_STAGES];
    voc_bigvgan_t bigvgan;

    /* Scratch buffers (ping-pong) */
    float *buf_a;
    float *buf_b;
    size_t buf_cap;             /* floats per buffer */

    /* Preallocated scratch for conv/resunit/convnext operations */
    float *scratch;
    size_t scratch_cap;         /* floats */

    /* Transformer scratch */
    float *xfmr_q, *xfmr_k, *xfmr_v;
    float *xfmr_attn_out, *xfmr_proj_out;
    float *xfmr_norm_buf;
    float *xfmr_gate_up;
    float *xfmr_ffn_out;
    float *rope_cos, *rope_sin;
    int xfmr_buf_cap;          /* max T allocated for transformer buffers */

    /* Safetensors handle (for the Tokenizer-12Hz model) */
    multi_safetensors_t *safetensors;

    int verbose;
} tts_vocoder_ctx_t;

/* ========================================================================
 * Per-stage timing (optional, for benchmarking)
 * ======================================================================== */

typedef struct {
    double rvq_ms;
    double preconv_ms;
    double xfmr_ms;
    double upsample_ms[VOC_UPSAMPLE_STAGES];
    double bigvgan_init_ms;
    double bigvgan_block_ms[VOC_BIGVGAN_NUM_BLOCKS];
    double bigvgan_tconv_ms[VOC_BIGVGAN_NUM_BLOCKS];  /* snake + conv_transpose */
    double bigvgan_res_ms[VOC_BIGVGAN_NUM_BLOCKS];     /* 3x resunit */
    double final_ms;
    double total_ms;
} voc_timing_t;

/* ========================================================================
 * API
 * ======================================================================== */

/* Initialize vocoder from Qwen3-TTS-Tokenizer-12Hz model directory.
 * The directory must contain model.safetensors with decoder.* weights.
 * Returns 0 on success. */
int tts_vocoder_init(tts_vocoder_ctx_t *ctx, const char *tokenizer12hz_dir,
                     int verbose);

/* Free all vocoder resources. */
void tts_vocoder_free(tts_vocoder_ctx_t *ctx);

/* Run vocoder: codes [n_steps, 16] int64 -> audio float samples.
 * If timing is non-NULL, populates per-stage timing data.
 * Returns malloc'd float array, sets *out_n_samples.
 * Caller must free the returned array. */
float *tts_vocoder_run(tts_vocoder_ctx_t *ctx, const int64_t *codes,
                       int n_steps, int *out_n_samples,
                       voc_timing_t *timing);

/* ========================================================================
 * Internal operations (implemented in tts_vocoder_ops.c)
 * ======================================================================== */

/* Causal Conv1d: left-pad by dilation*(kernel-1), then conv.
 * in: [channels_in, T], out: [channels_out, T]
 * weight: [channels_out, channels_in, kernel] (or [ch_out, 1, kernel] for depthwise)
 * bias: [channels_out] or NULL
 * For groups > 1, implements depthwise (groups == channels_in == channels_out). */
void voc_conv1d_causal(float *out, const float *in,
                       const float *weight, const float *bias,
                       int c_in, int c_out, int T, int kernel, int dilation,
                       int groups, float *scratch);

/* Causal ConvTranspose1d: GEMM + col2im (with BLAS) or scatter-add (without).
 * in: [channels_in, T], out: [channels_out, T*stride]
 * weight: [channels_in, channels_out, kernel]
 * kernel = 2*stride for BigVGAN blocks, kernel = stride for upsample stages.
 * scratch must hold at least c_out * kernel * T floats (GEMM cols buffer). */
void voc_conv_transpose1d(float *out, const float *in,
                          const float *weight, const float *bias,
                          int c_in, int c_out, int T, int kernel, int stride,
                          float *scratch);

/* SnakeBeta activation: x + (1/exp(beta)) * sin^2(exp(alpha) * x)
 * In-place on x [channels, T]. alpha, beta: [channels] (pre-exponentiated at load). */
void voc_snake_beta(float *x, const float *exp_alpha, const float *inv_exp_beta,
                    int channels, int T);

/* LayerNorm over channel dimension (for ConvNeXt: input is [C, T], normalize each time step).
 * Transposes internally: for each t, normalize x[:, t]. */
void voc_layer_norm_channels(float *x, const float *weight, const float *bias,
                             int channels, int T, float eps);

/* GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2))) */
void voc_gelu(float *x, int n);

/* ========================================================================
 * Internal: pre-transformer (implemented in tts_vocoder_xfmr.c)
 * ======================================================================== */

/* Run 8-layer pre-transformer.
 * in: [1024, T] (channels-first from conv), out: [1024, T]
 * Internally transposes to [T, 1024] for attention, then back. */
void voc_pre_transformer(tts_vocoder_ctx_t *ctx, float *out, const float *in, int T);

#endif /* TTS_VOCODER_H */
