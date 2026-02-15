/*
 * tts_vocoder.c - Native C vocoder for Qwen3-TTS Tokenizer-12Hz
 *
 * Weight loading from safetensors, RVQ decode (codebook lookups + 1x1 conv
 * projections), buffer management, and full vocoder pipeline orchestration.
 *
 * Pipeline: codes [T, 16] -> RVQ decode [512, T] -> pre_conv [1024, T] ->
 *           pre_transformer [1024, T] -> upsample [1024, 4T] ->
 *           BigVGAN [1, 1920T] -> clamp -> audio
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

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* ========================================================================
 * Helpers
 * ======================================================================== */

static float *voc_load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "vocoder: missing weight '%s'\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static float *voc_try_load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

/* Load codebook embeddings with cluster normalization:
 * embedding = embedding_sum / clamp(cluster_usage, 1e-7)
 * Returns [codebook_size, dim] f32 array. */
static float *load_codebook(multi_safetensors_t *ms, const char *sum_name,
                             const char *usage_name, int codebook_size, int dim) {
    float *emb_sum = voc_load_f32(ms, sum_name);
    float *usage = voc_load_f32(ms, usage_name);
    if (!emb_sum || !usage) {
        free(emb_sum);
        free(usage);
        return NULL;
    }

    float *emb = emb_sum; /* reuse the allocation */
    for (int i = 0; i < codebook_size; i++) {
        float u = usage[i];
        if (u < 1e-7f) u = 1e-7f;
        float inv_u = 1.0f / u;
        for (int d = 0; d < dim; d++) {
            emb[i * dim + d] *= inv_u;
        }
    }
    free(usage);
    return emb;
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static int load_rvq_weights(tts_vocoder_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    char name[512];

    /* First codebook (rvq_first): [2048, 256] */
    ctx->codebooks[0].embeddings = load_codebook(ms,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
        VOC_CODEBOOK_SIZE, VOC_CODEBOOK_DIM);
    if (!ctx->codebooks[0].embeddings) return -1;

    ctx->codebooks[0].proj_weight = voc_load_f32(ms,
        "decoder.quantizer.rvq_first.output_proj.weight");
    if (!ctx->codebooks[0].proj_weight) return -1;

    /* Rest codebooks (rvq_rest): 15 codebooks, each [2048, 256] */
    for (int i = 0; i < 15; i++) {
        int cb_idx = i + 1;
        snprintf(name, sizeof(name),
            "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.embedding_sum", i);
        char usage_name[512];
        snprintf(usage_name, sizeof(usage_name),
            "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.cluster_usage", i);
        ctx->codebooks[cb_idx].embeddings = load_codebook(ms, name, usage_name,
            VOC_CODEBOOK_SIZE, VOC_CODEBOOK_DIM);
        if (!ctx->codebooks[cb_idx].embeddings) return -1;
    }

    /* Shared output_proj for rest codebooks */
    float *rest_proj = voc_load_f32(ms, "decoder.quantizer.rvq_rest.output_proj.weight");
    if (!rest_proj) return -1;
    ctx->codebooks[1].proj_weight = rest_proj;
    for (int i = 2; i < VOC_NUM_CODEBOOKS; i++) {
        ctx->codebooks[i].proj_weight = rest_proj; /* shared, only free once */
    }

    return 0;
}

static int load_pre_conv_weights(tts_vocoder_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    ctx->pre_conv_weight = voc_load_f32(ms, "decoder.pre_conv.conv.weight");
    ctx->pre_conv_bias = voc_load_f32(ms, "decoder.pre_conv.conv.bias");
    return (ctx->pre_conv_weight && ctx->pre_conv_bias) ? 0 : -1;
}

static int load_transformer_weights(tts_vocoder_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    voc_pre_transformer_t *xf = &ctx->xfmr;
    char name[512];

    xf->input_proj = voc_load_f32(ms, "decoder.pre_transformer.input_proj.weight");
    xf->input_proj_bias = voc_try_load_f32(ms, "decoder.pre_transformer.input_proj.bias");
    xf->output_proj = voc_load_f32(ms, "decoder.pre_transformer.output_proj.weight");
    xf->output_proj_bias = voc_try_load_f32(ms, "decoder.pre_transformer.output_proj.bias");
    xf->final_norm = voc_load_f32(ms, "decoder.pre_transformer.norm.weight");
    if (!xf->input_proj || !xf->output_proj || !xf->final_norm) return -1;

    for (int i = 0; i < VOC_XFMR_LAYERS; i++) {
        voc_xfmr_layer_t *l = &xf->layers[i];

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.q_proj.weight", i);
        l->wq = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.k_proj.weight", i);
        l->wk = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.v_proj.weight", i);
        l->wv = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.o_proj.weight", i);
        l->wo = voc_load_f32(ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.input_layernorm.weight", i);
        l->input_norm = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.post_attention_layernorm.weight", i);
        l->post_attn_norm = voc_load_f32(ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.gate_proj.weight", i);
        l->gate_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.up_proj.weight", i);
        l->up_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.down_proj.weight", i);
        l->down_weight = voc_load_f32(ms, name);

        /* Layer scales: self_attn_layer_scale.scale and mlp_layer_scale.scale */
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn_layer_scale.scale", i);
        l->attn_layer_scale = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp_layer_scale.scale", i);
        l->mlp_layer_scale = voc_load_f32(ms, name);

        if (!l->wq || !l->wk || !l->wv || !l->wo ||
            !l->input_norm || !l->post_attn_norm ||
            !l->gate_weight || !l->up_weight || !l->down_weight ||
            !l->attn_layer_scale || !l->mlp_layer_scale) {
            fprintf(stderr, "vocoder: missing transformer layer %d weight\n", i);
            return -1;
        }
    }

    return 0;
}

static int load_upsample_weights(tts_vocoder_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    char name[512];

    for (int s = 0; s < VOC_UPSAMPLE_STAGES; s++) {
        voc_upsample_stage_t *us = &ctx->upsample[s];
        us->stride = 2;

        /* Transposed convolution */
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.weight", s);
        us->tconv_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.bias", s);
        us->tconv_bias = voc_load_f32(ms, name);
        if (!us->tconv_weight || !us->tconv_bias) return -1;

        /* ConvNeXt block -- actual names: dwconv, pwconv1, pwconv2 */
        voc_convnext_t *cn = &us->convnext;
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.weight", s);
        cn->dw_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.bias", s);
        cn->dw_bias = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.weight", s);
        cn->norm_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.bias", s);
        cn->norm_bias = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.weight", s);
        cn->pw1_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.bias", s);
        cn->pw1_bias = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.weight", s);
        cn->pw2_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.bias", s);
        cn->pw2_bias = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.gamma", s);
        cn->gamma = voc_load_f32(ms, name);

        if (!cn->dw_weight || !cn->dw_bias || !cn->norm_weight || !cn->norm_bias ||
            !cn->pw1_weight || !cn->pw1_bias || !cn->pw2_weight || !cn->pw2_bias ||
            !cn->gamma) return -1;
    }

    return 0;
}

/* Pre-compute exp(alpha) and 1/exp(beta) for SnakeBeta at load time */
static void precompute_snake(float *alpha, float *beta, int n) {
    for (int i = 0; i < n; i++) {
        alpha[i] = expf(alpha[i]);
        beta[i] = 1.0f / expf(beta[i]);
    }
}

static int load_bigvgan_weights(tts_vocoder_ctx_t *ctx) {
    multi_safetensors_t *ms = ctx->safetensors;
    voc_bigvgan_t *bg = &ctx->bigvgan;
    char name[512];

    /* decoder.decoder.0 = init CausalConv1d(1024, 1536, k=7) */
    bg->init_weight = voc_load_f32(ms, "decoder.decoder.0.conv.weight");
    bg->init_bias = voc_load_f32(ms, "decoder.decoder.0.conv.bias");
    if (!bg->init_weight || !bg->init_bias) return -1;

    /* BigVGAN blocks: decoder.decoder.{1,2,3,4}
     * Naming: .block.0 = snake, .block.1 = transconv,
     *         .block.{2,3,4} = resunits (act1/conv1/act2/conv2) */
    static const int rates[] = { 8, 5, 4, 3 };
    static const int channels[] = { 1536, 768, 384, 192, 96 };

    for (int b = 0; b < VOC_BIGVGAN_NUM_BLOCKS; b++) {
        voc_bigvgan_block_t *blk = &bg->blocks[b];
        int dec_idx = b + 1;
        blk->rate = rates[b];
        blk->in_ch = channels[b];
        blk->out_ch = channels[b + 1];

        /* SnakeBeta: decoder.decoder.{N}.block.0 */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.alpha", dec_idx);
        blk->snake_alpha = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.beta", dec_idx);
        blk->snake_beta = voc_load_f32(ms, name);
        if (!blk->snake_alpha || !blk->snake_beta) return -1;
        precompute_snake(blk->snake_alpha, blk->snake_beta, blk->in_ch);

        /* Transposed conv: decoder.decoder.{N}.block.1 */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.weight", dec_idx);
        blk->tconv_weight = voc_load_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.bias", dec_idx);
        blk->tconv_bias = voc_load_f32(ms, name);
        if (!blk->tconv_weight || !blk->tconv_bias) return -1;

        /* 3 ResUnits: decoder.decoder.{N}.block.{2,3,4}
         * Within each: act1 (snake1), conv1, act2 (snake2), conv2 */
        static const int dilations[] = { 1, 3, 9 };
        for (int r = 0; r < VOC_BIGVGAN_RESUNITS; r++) {
            voc_resunit_t *ru = &blk->resunits[r];
            ru->dilation = dilations[r];
            int ch = blk->out_ch;
            int ru_idx = r + 2; /* block.2, block.3, block.4 */

            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.alpha", dec_idx, ru_idx);
            ru->snake1_alpha = voc_load_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.beta", dec_idx, ru_idx);
            ru->snake1_beta = voc_load_f32(ms, name);
            if (!ru->snake1_alpha || !ru->snake1_beta) return -1;
            precompute_snake(ru->snake1_alpha, ru->snake1_beta, ch);

            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.weight", dec_idx, ru_idx);
            ru->conv1_weight = voc_load_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.bias", dec_idx, ru_idx);
            ru->conv1_bias = voc_load_f32(ms, name);

            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.alpha", dec_idx, ru_idx);
            ru->snake2_alpha = voc_load_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.beta", dec_idx, ru_idx);
            ru->snake2_beta = voc_load_f32(ms, name);
            if (!ru->snake2_alpha || !ru->snake2_beta) return -1;
            precompute_snake(ru->snake2_alpha, ru->snake2_beta, ch);

            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.weight", dec_idx, ru_idx);
            ru->conv2_weight = voc_load_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.bias", dec_idx, ru_idx);
            ru->conv2_bias = voc_load_f32(ms, name);

            if (!ru->conv1_weight || !ru->conv1_bias ||
                !ru->conv2_weight || !ru->conv2_bias) return -1;
        }
    }

    /* Final SnakeBeta + Conv */
    bg->final_snake_alpha = voc_load_f32(ms, "decoder.decoder.5.alpha");
    bg->final_snake_beta = voc_load_f32(ms, "decoder.decoder.5.beta");
    if (!bg->final_snake_alpha || !bg->final_snake_beta) return -1;
    precompute_snake(bg->final_snake_alpha, bg->final_snake_beta, 96);

    bg->final_weight = voc_load_f32(ms, "decoder.decoder.6.conv.weight");
    bg->final_bias = voc_load_f32(ms, "decoder.decoder.6.conv.bias");
    if (!bg->final_weight || !bg->final_bias) return -1;

    return 0;
}

/* ========================================================================
 * Buffer Management
 * ======================================================================== */

static void ensure_buffers(tts_vocoder_ctx_t *ctx, int T) {
    /* Peak: BigVGAN block 3 output [96, 1920T] = 184320*T floats per buffer. */
    size_t needed = (size_t)184320 * T;
    if (needed > ctx->buf_cap) {
        free(ctx->buf_a);
        free(ctx->buf_b);
        ctx->buf_a = (float *)malloc(needed * sizeof(float));
        ctx->buf_b = (float *)malloc(needed * sizeof(float));
        ctx->buf_cap = needed;
        if (!ctx->buf_a || !ctx->buf_b) {
            fprintf(stderr, "vocoder: failed to allocate buffers (%zu floats)\n", needed);
        }
    }

    /* Compute peak scratch size by simulating T progression through all stages.
     * Scratch is used one stage at a time (never nested), so we track the max. */
    static const int bigvgan_rates[] = { 8, 5, 4, 3 };
    static const int bigvgan_channels[] = { 1536, 768, 384, 192, 96 };
    static const int dilations[] = { 1, 3, 9 };

    size_t max_scratch = 0;
    size_t s;
    int T_cur = T;

    /* Conv1d scratch: padded[c_in, T_padded] + extra.
     * With USE_BLAS, conv1d uses implicit GEMM (extra = w_slice[c_out*c_in]) when
     * im2col would exceed 10M floats, otherwise im2col (extra = cols[c_in*kernel*T]).
     * Without USE_BLAS: always im2col. */
#ifdef USE_BLAS
#define VOC_CONV_SCRATCH(c_in, c_out, T, k, T_pad) do { \
        size_t im2col_n = (size_t)(c_in) * (k) * (T); \
        s = (size_t)(c_in) * (T_pad); \
        if (im2col_n <= 10000000) \
            s += im2col_n; \
        else \
            s += (size_t)(c_out) * (c_in); \
    } while (0)
#else
#define VOC_CONV_SCRATCH(c_in, c_out, T, k, T_pad) do { \
        s = (size_t)(c_in) * (T_pad) \
          + (size_t)(c_in) * (k) * (T); \
    } while (0)
#endif

    /* Pre-conv: c_in=512, c_out=1024, k=3 */
    VOC_CONV_SCRATCH(VOC_PRE_CONV_IN, VOC_PRE_CONV_OUT, T_cur,
                     VOC_PRE_CONV_KERNEL, T_cur + VOC_PRE_CONV_KERNEL - 1);
    if (s > max_scratch) max_scratch = s;

    /* Upsample ConvTranspose cols: [c_out * kernel, T_cur] = [1024*2, T_cur] */
    for (int i = 0; i < VOC_UPSAMPLE_STAGES; i++) {
        s = (size_t)VOC_UPSAMPLE_CHANNELS * 2 * T_cur; /* cols for transpose */
        if (s > max_scratch) max_scratch = s;
        T_cur *= 2; /* stride=2 */
        /* ConvNeXt: residual + depthwise_scratch + pointwise_scratch */
        s = (size_t)VOC_UPSAMPLE_CHANNELS * T_cur * 6;
        if (s > max_scratch) max_scratch = s;
    }

    /* BigVGAN init conv: c_in=1024, c_out=1536, k=7 */
    VOC_CONV_SCRATCH(1024, 1536, T_cur, 7, T_cur + 6);
    if (s > max_scratch) max_scratch = s;

    /* BigVGAN blocks: ConvTranspose cols + ResUnit scratch */
    for (int b = 0; b < VOC_BIGVGAN_NUM_BLOCKS; b++) {
        int tconv_kernel = 2 * bigvgan_rates[b];
        int ch_in = bigvgan_channels[b];
        int ch_out = bigvgan_channels[b + 1];

        /* ConvTranspose cols: [ch_out * tconv_kernel, T_cur] */
        s = (size_t)ch_out * tconv_kernel * T_cur;
        if (s > max_scratch) max_scratch = s;

        T_cur = (T_cur + 1) * bigvgan_rates[b] - tconv_kernel;
        int ch = ch_out;
        for (int r = 0; r < VOC_BIGVGAN_RESUNITS; r++) {
            /* ResUnit scratch: residual[ch*T] + conv_out[ch*T] + conv_scratch */
            int T_padded = T_cur + dilations[r] * 6;
            size_t conv_scratch;
            VOC_CONV_SCRATCH(ch, ch, T_cur, VOC_BIGVGAN_RES_KERNEL, T_padded);
            conv_scratch = s;
            s = (size_t)ch * T_cur * 2 + conv_scratch;
            if (s > max_scratch) max_scratch = s;
        }
    }

    /* Final conv: c_in=96, c_out=1, k=7 */
    VOC_CONV_SCRATCH(96, 1, T_cur, 7, T_cur + 6);
    if (s > max_scratch) max_scratch = s;

#undef VOC_CONV_SCRATCH

    if (max_scratch > ctx->scratch_cap) {
        free(ctx->scratch);
        ctx->scratch = (float *)malloc(max_scratch * sizeof(float));
        ctx->scratch_cap = max_scratch;
        if (!ctx->scratch) {
            fprintf(stderr, "vocoder: failed to allocate scratch (%zu floats)\n", max_scratch);
        }
    }
}

/* ========================================================================
 * RVQ Decode
 * ======================================================================== */

static void rvq_decode(tts_vocoder_ctx_t *ctx, const int64_t *codes,
                        int T, float *out) {
    memset(out, 0, (size_t)VOC_RVQ_OUT_DIM * T * sizeof(float));

    float embed_buf[VOC_CODEBOOK_DIM]; /* 256 */
    float proj_buf[VOC_RVQ_OUT_DIM];   /* 512 */

    for (int t = 0; t < T; t++) {
        for (int cb = 0; cb < VOC_NUM_CODEBOOKS; cb++) {
            int code = (int)codes[t * VOC_NUM_CODEBOOKS + cb];

            const float *emb = ctx->codebooks[cb].embeddings +
                               (size_t)code * VOC_CODEBOOK_DIM;
            memcpy(embed_buf, emb, VOC_CODEBOOK_DIM * sizeof(float));

            const float *proj = ctx->codebooks[cb].proj_weight;
            for (int o = 0; o < VOC_RVQ_OUT_DIM; o++) {
                float sum = 0.0f;
                for (int d = 0; d < VOC_CODEBOOK_DIM; d++) {
                    sum += proj[o * VOC_CODEBOOK_DIM + d] * embed_buf[d];
                }
                proj_buf[o] = sum;
            }

            for (int o = 0; o < VOC_RVQ_OUT_DIM; o++) {
                out[(size_t)o * T + t] += proj_buf[o];
            }
        }
    }
}

/* ========================================================================
 * ConvNeXt Block
 * ======================================================================== */

static void run_convnext(tts_vocoder_ctx_t *ctx, float *x, int channels, int T,
                          const voc_convnext_t *cn, float *scratch) {
    size_t cT = (size_t)channels * T;
    float *residual = scratch;
    memcpy(residual, x, cT * sizeof(float));
    float *scratch2 = scratch + cT;

    /* Depthwise causal conv (groups=channels, k=7) */
    voc_conv1d_causal(x, x, cn->dw_weight, cn->dw_bias,
                       channels, channels, T, VOC_CONVNEXT_KERNEL, 1,
                       channels, scratch2);

    /* LayerNorm across channels for each time step */
    voc_layer_norm_channels(x, cn->norm_weight, cn->norm_bias, channels, T, 1e-6f);

    /* Pointwise conv1 (up): [channels, T] -> [channels*4, T] */
    int ch4 = channels * 4;
    float *pw1_out = scratch2;
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ch4, T, channels,
                1.0f, cn->pw1_weight, channels, x, T,
                0.0f, pw1_out, T);
#else
    for (int o = 0; o < ch4; o++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int c = 0; c < channels; c++) {
                sum += cn->pw1_weight[o * channels + c] * x[(size_t)c * T + t];
            }
            pw1_out[(size_t)o * T + t] = sum;
        }
    }
#endif
    for (int o = 0; o < ch4; o++) {
        float b = cn->pw1_bias[o];
        for (int t = 0; t < T; t++) pw1_out[(size_t)o * T + t] += b;
    }

    voc_gelu(pw1_out, ch4 * T);

    /* Pointwise conv2 (down): [ch4, T] -> [channels, T] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                channels, T, ch4,
                1.0f, cn->pw2_weight, ch4, pw1_out, T,
                0.0f, x, T);
#else
    for (int o = 0; o < channels; o++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int c = 0; c < ch4; c++) {
                sum += cn->pw2_weight[o * ch4 + c] * pw1_out[(size_t)c * T + t];
            }
            x[(size_t)o * T + t] = sum;
        }
    }
#endif
    for (int o = 0; o < channels; o++) {
        float b = cn->pw2_bias[o];
        for (int t = 0; t < T; t++) x[(size_t)o * T + t] += b;
    }

    /* LayerScale */
    for (int c = 0; c < channels; c++) {
        float g = cn->gamma[c];
        for (int t = 0; t < T; t++) x[(size_t)c * T + t] *= g;
    }

    /* Add residual */
    for (size_t i = 0; i < cT; i++) x[i] += residual[i];
}

/* ========================================================================
 * BigVGAN ResUnit
 * ======================================================================== */

static void run_resunit(float *x, const voc_resunit_t *ru, int channels, int T,
                         float *scratch) {
    size_t cT = (size_t)channels * T;

    float *residual = scratch;
    memcpy(residual, x, cT * sizeof(float));
    float *scratch2 = scratch + cT;

    voc_snake_beta(x, ru->snake1_alpha, ru->snake1_beta, channels, T);

    float *conv_out = scratch2;
    float *conv_scratch = scratch2 + cT;
    voc_conv1d_causal(conv_out, x, ru->conv1_weight, ru->conv1_bias,
                       channels, channels, T, VOC_BIGVGAN_RES_KERNEL,
                       ru->dilation, 1, conv_scratch);

    voc_snake_beta(conv_out, ru->snake2_alpha, ru->snake2_beta, channels, T);

    /* Conv1d k=1 (pointwise): weight [channels, channels, 1] -> GEMM */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                channels, T, channels,
                1.0f, ru->conv2_weight, channels, conv_out, T,
                0.0f, x, T);
#else
    for (int o = 0; o < channels; o++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int c = 0; c < channels; c++) {
                sum += ru->conv2_weight[o * channels + c] * conv_out[(size_t)c * T + t];
            }
            x[(size_t)o * T + t] = sum;
        }
    }
#endif
    if (ru->conv2_bias) {
        for (int o = 0; o < channels; o++) {
            float b = ru->conv2_bias[o];
            for (int t = 0; t < T; t++) x[(size_t)o * T + t] += b;
        }
    }

    for (size_t i = 0; i < cT; i++) x[i] += residual[i];
}

/* ========================================================================
 * Full Vocoder Pipeline
 * ======================================================================== */

float *tts_vocoder_run(tts_vocoder_ctx_t *ctx, const int64_t *codes,
                        int n_steps, int *out_n_samples,
                        voc_timing_t *timing) {
    *out_n_samples = 0;
    int T = n_steps;

    LARGE_INTEGER freq, t_start, t_prev, t_now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);
    t_prev = t_start;

    if (timing) memset(timing, 0, sizeof(*timing));

    ensure_buffers(ctx, T);

    if (ctx->verbose) {
        printf("  vocoder native: %d codec frames\n", T);
        fflush(stdout);
    }

    /* Stage 1: RVQ decode -> [512, T] */
    float *cur = ctx->buf_a;

    rvq_decode(ctx, codes, T, cur);

    QueryPerformanceCounter(&t_now);
    if (timing) {
        timing->rvq_ms = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }
    if (ctx->verbose) {
        double ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("    RVQ decode: [512, %d] (%.0f ms)\n", T, ms);
        fflush(stdout);
    }
    t_prev = t_now;

    /* Stage 2: Pre-conv CausalConv1d(512, 1024, k=3) -> [1024, T] */
    float *next = ctx->buf_b;
    float *scratch = ctx->scratch;
    voc_conv1d_causal(next, cur, ctx->pre_conv_weight, ctx->pre_conv_bias,
                       VOC_PRE_CONV_IN, VOC_PRE_CONV_OUT, T,
                       VOC_PRE_CONV_KERNEL, 1, 1, scratch);
    cur = next;

    QueryPerformanceCounter(&t_now);
    if (timing) {
        timing->preconv_ms = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }
    if (ctx->verbose) {
        double ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("    pre_conv: [1024, %d] (%.0f ms)\n", T, ms);
        fflush(stdout);
    }
    t_prev = t_now;

    /* Stage 3: Pre-transformer -> [1024, T] (no external residual) */
    next = ctx->buf_a;
    voc_pre_transformer(ctx, next, cur, T);
    cur = next;

    QueryPerformanceCounter(&t_now);
    if (timing) {
        timing->xfmr_ms = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }
    if (ctx->verbose) {
        double ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("    pre_transformer: [1024, %d] (%.0f ms)\n", T, ms);
        fflush(stdout);
    }
    t_prev = t_now;

    /* Stage 4: ConvNeXt Upsample (2 stages, each 2x) -> [1024, 4T] */
    for (int s = 0; s < VOC_UPSAMPLE_STAGES; s++) {
        voc_upsample_stage_t *us = &ctx->upsample[s];
        int T_out = T * us->stride;

        next = (cur == ctx->buf_a) ? ctx->buf_b : ctx->buf_a;
        voc_conv_transpose1d(next, cur, us->tconv_weight, us->tconv_bias,
                              VOC_UPSAMPLE_CHANNELS, VOC_UPSAMPLE_CHANNELS,
                              T, us->stride, us->stride, scratch);
        T = T_out;
        cur = next;

        run_convnext(ctx, cur, VOC_UPSAMPLE_CHANNELS, T, &us->convnext, scratch);

        QueryPerformanceCounter(&t_now);
        if (timing) {
            timing->upsample_ms[s] = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
        }
        if (ctx->verbose) {
            printf("    upsample%d: [1024, %d]\n", s, T);
            fflush(stdout);
        }
        t_prev = t_now;
    }

    if (ctx->verbose) {
        double ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("    upsample: [1024, %d] (%.0f ms)\n", T, ms);
        fflush(stdout);
    }

    /* Stage 5: BigVGAN decoder */
    /* Init conv: CausalConv1d(1024, 1536, k=7) */
    {
        next = (cur == ctx->buf_a) ? ctx->buf_b : ctx->buf_a;
        voc_conv1d_causal(next, cur, ctx->bigvgan.init_weight, ctx->bigvgan.init_bias,
                           1024, VOC_BIGVGAN_INIT_CH, T, 7, 1, 1, scratch);
        cur = next;

        QueryPerformanceCounter(&t_now);
        if (timing) {
            timing->bigvgan_init_ms = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
        }
        if (ctx->verbose) {
            printf("    bigvgan_init: [1536, %d]\n", T);
            fflush(stdout);
        }
        t_prev = t_now;
    }

    /* 4 BigVGAN blocks */
    for (int b = 0; b < VOC_BIGVGAN_NUM_BLOCKS; b++) {
        voc_bigvgan_block_t *blk = &ctx->bigvgan.blocks[b];
        int tconv_kernel = 2 * blk->rate;
        /* ConvTranspose1d trims (kernel-stride) from each side: T_out = (T-1)*rate */
        int T_out = (T + 1) * blk->rate - tconv_kernel;

        voc_snake_beta(cur, blk->snake_alpha, blk->snake_beta, blk->in_ch, T);

        next = (cur == ctx->buf_a) ? ctx->buf_b : ctx->buf_a;
        voc_conv_transpose1d(next, cur, blk->tconv_weight, blk->tconv_bias,
                              blk->in_ch, blk->out_ch, T, tconv_kernel, blk->rate,
                              scratch);
        T = T_out;
        cur = next;

        LARGE_INTEGER t_mid;
        QueryPerformanceCounter(&t_mid);

        for (int r = 0; r < VOC_BIGVGAN_RESUNITS; r++) {
            run_resunit(cur, &blk->resunits[r], blk->out_ch, T, scratch);
        }

        QueryPerformanceCounter(&t_now);
        if (timing) {
            timing->bigvgan_block_ms[b] = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
            timing->bigvgan_tconv_ms[b] = (double)(t_mid.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
            timing->bigvgan_res_ms[b] = (double)(t_now.QuadPart - t_mid.QuadPart) * 1000.0 / (double)freq.QuadPart;
        }
        if (ctx->verbose) {
            double ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
            printf("    bigvgan block %d: [%d, %d] (%.0f ms)\n", b, blk->out_ch, T, ms);
            fflush(stdout);
        }
        t_prev = t_now;
    }

    /* Final: SnakeBeta(96) -> CausalConv1d(96, 1, k=7) -> clamp(-1, 1) */
    voc_snake_beta(cur, ctx->bigvgan.final_snake_alpha, ctx->bigvgan.final_snake_beta,
                    96, T);

    next = (cur == ctx->buf_a) ? ctx->buf_b : ctx->buf_a;
    voc_conv1d_causal(next, cur, ctx->bigvgan.final_weight, ctx->bigvgan.final_bias,
                       96, 1, T, 7, 1, 1, scratch);

    for (int i = 0; i < T; i++) {
        if (next[i] > 1.0f) next[i] = 1.0f;
        if (next[i] < -1.0f) next[i] = -1.0f;
    }

    QueryPerformanceCounter(&t_now);
    if (timing) {
        timing->final_ms = (double)(t_now.QuadPart - t_prev.QuadPart) * 1000.0 / (double)freq.QuadPart;
        timing->total_ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }
    if (ctx->verbose) {
        double ms = (double)(t_now.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("    final: %d samples (%.0f ms total)\n", T, ms);
        fflush(stdout);
    }

    int n_samples = T;
    float *audio = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!audio) return NULL;
    memcpy(audio, next, (size_t)n_samples * sizeof(float));

    *out_n_samples = n_samples;
    return audio;
}

/* ========================================================================
 * Init / Free
 * ======================================================================== */

int tts_vocoder_init(tts_vocoder_ctx_t *ctx, const char *tokenizer12hz_dir,
                      int verbose) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->verbose = verbose;

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    if (verbose) printf("vocoder: loading weights from %s\n", tokenizer12hz_dir);

    ctx->safetensors = multi_safetensors_open(tokenizer12hz_dir);
    if (!ctx->safetensors) {
        fprintf(stderr, "vocoder: failed to open safetensors in %s\n", tokenizer12hz_dir);
        return -1;
    }

    if (load_rvq_weights(ctx) != 0) goto fail;
    if (verbose) printf("vocoder: RVQ codebooks loaded (16 codebooks)\n");

    if (load_pre_conv_weights(ctx) != 0) goto fail;
    if (verbose) printf("vocoder: pre-conv loaded\n");

    if (load_transformer_weights(ctx) != 0) goto fail;
    if (verbose) printf("vocoder: pre-transformer loaded (8 layers)\n");

    if (load_upsample_weights(ctx) != 0) goto fail;
    if (verbose) printf("vocoder: upsample loaded (2 stages)\n");

    if (load_bigvgan_weights(ctx) != 0) goto fail;
    if (verbose) printf("vocoder: BigVGAN loaded (4 blocks)\n");

    QueryPerformanceCounter(&t_end);
    double ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
    if (verbose) printf("vocoder: all weights loaded in %.0f ms\n", ms);

    return 0;

fail:
    tts_vocoder_free(ctx);
    return -1;
}

void tts_vocoder_free(tts_vocoder_ctx_t *ctx) {
    for (int i = 0; i < VOC_NUM_CODEBOOKS; i++) {
        free(ctx->codebooks[i].embeddings);
    }
    free(ctx->codebooks[0].proj_weight);
    if (VOC_NUM_CODEBOOKS > 1) free(ctx->codebooks[1].proj_weight);

    free(ctx->pre_conv_weight);
    free(ctx->pre_conv_bias);

    free(ctx->xfmr.input_proj);
    free(ctx->xfmr.input_proj_bias);
    free(ctx->xfmr.output_proj);
    free(ctx->xfmr.output_proj_bias);
    free(ctx->xfmr.final_norm);
    for (int i = 0; i < VOC_XFMR_LAYERS; i++) {
        voc_xfmr_layer_t *l = &ctx->xfmr.layers[i];
        free(l->wq); free(l->wk); free(l->wv); free(l->wo);
        free(l->input_norm); free(l->post_attn_norm);
        free(l->gate_weight); free(l->up_weight); free(l->down_weight);
        free(l->attn_layer_scale); free(l->mlp_layer_scale);
    }

    for (int s = 0; s < VOC_UPSAMPLE_STAGES; s++) {
        voc_upsample_stage_t *us = &ctx->upsample[s];
        free(us->tconv_weight); free(us->tconv_bias);
        voc_convnext_t *cn = &us->convnext;
        free(cn->dw_weight); free(cn->dw_bias);
        free(cn->norm_weight); free(cn->norm_bias);
        free(cn->pw1_weight); free(cn->pw1_bias);
        free(cn->pw2_weight); free(cn->pw2_bias);
        free(cn->gamma);
    }

    free(ctx->bigvgan.init_weight); free(ctx->bigvgan.init_bias);
    for (int b = 0; b < VOC_BIGVGAN_NUM_BLOCKS; b++) {
        voc_bigvgan_block_t *blk = &ctx->bigvgan.blocks[b];
        free(blk->snake_alpha); free(blk->snake_beta);
        free(blk->tconv_weight); free(blk->tconv_bias);
        for (int r = 0; r < VOC_BIGVGAN_RESUNITS; r++) {
            voc_resunit_t *ru = &blk->resunits[r];
            free(ru->snake1_alpha); free(ru->snake1_beta);
            free(ru->conv1_weight); free(ru->conv1_bias);
            free(ru->snake2_alpha); free(ru->snake2_beta);
            free(ru->conv2_weight); free(ru->conv2_bias);
        }
    }
    free(ctx->bigvgan.final_snake_alpha); free(ctx->bigvgan.final_snake_beta);
    free(ctx->bigvgan.final_weight); free(ctx->bigvgan.final_bias);

    free(ctx->buf_a); free(ctx->buf_b);
    free(ctx->scratch);
    free(ctx->xfmr_q); free(ctx->xfmr_k); free(ctx->xfmr_v);
    free(ctx->xfmr_attn_out); free(ctx->xfmr_proj_out);
    free(ctx->xfmr_norm_buf); free(ctx->xfmr_gate_up);
    free(ctx->xfmr_ffn_out);
    free(ctx->rope_cos); free(ctx->rope_sin);

    if (ctx->safetensors) multi_safetensors_close(ctx->safetensors);

    memset(ctx, 0, sizeof(*ctx));
}
