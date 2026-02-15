/*
 * tts_speaker_enc.c - ECAPA-TDNN speaker encoder for Qwen3-TTS
 *
 * Computes 1024-dim speaker embeddings from 128-band mel spectrograms.
 * Uses same-padded Conv1d (not causal like the vocoder).
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_speaker_enc.h"
#include "qwen_asr_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

/* ========================================================================
 * Same-padded Conv1d (different from vocoder's causal conv1d)
 * ======================================================================== */

/* Conv1d with same padding: pad = dilation * (kernel - 1) / 2 on each side.
 * in: [c_in, T], out: [c_out, T]
 * weight: [c_out, c_in, kernel]
 * scratch must hold at least c_in * T_padded + c_in * kernel * T floats. */
static void conv1d_same(float *out, const float *in,
                        const float *weight, const float *bias,
                        int c_in, int c_out, int T, int kernel, int dilation,
                        float *scratch) {
    int pad = dilation * (kernel - 1) / 2;
    int T_padded = T + 2 * pad;
    int patch_size = c_in * kernel;

    /* Pad input: [c_in, T_padded] (zero-pad both sides) */
    float *padded = scratch;
    for (int ch = 0; ch < c_in; ch++) {
        float *dst = padded + (size_t)ch * T_padded;
        memset(dst, 0, (size_t)pad * sizeof(float));
        memcpy(dst + pad, in + (size_t)ch * T, (size_t)T * sizeof(float));
        memset(dst + pad + T, 0, (size_t)pad * sizeof(float));
    }

#ifdef USE_BLAS
    /* im2col + SGEMM */
    float *cols = padded + (size_t)c_in * T_padded;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kernel; ki++) {
            int col_row = ic * kernel + ki;
            float *col_ptr = cols + (size_t)col_row * T;
            const float *in_ch = padded + (size_t)ic * T_padded;
            for (int t = 0; t < T; t++) {
                col_ptr[t] = in_ch[t + ki * dilation];
            }
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, T, patch_size,
                1.0f, weight, patch_size, cols, T,
                0.0f, out, T);
#else
    /* Naive fallback */
    for (int oc = 0; oc < c_out; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int ic = 0; ic < c_in; ic++) {
                const float *in_ch = padded + (size_t)ic * T_padded;
                for (int ki = 0; ki < kernel; ki++) {
                    sum += weight[oc * patch_size + ic * kernel + ki]
                         * in_ch[t + ki * dilation];
                }
            }
            out[(size_t)oc * T + t] = sum;
        }
    }
#endif

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + (size_t)oc * T;
            for (int t = 0; t < T; t++) {
                row[t] += b;
            }
        }
    }
}

/* Conv1d with k=1 (no padding needed, just SGEMM linear projection over time).
 * in: [c_in, T], out: [c_out, T], weight: [c_out, c_in] */
static void conv1d_k1(float *out, const float *in,
                      const float *weight, const float *bias,
                      int c_in, int c_out, int T) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, T, c_in,
                1.0f, weight, c_in, in, T,
                0.0f, out, T);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int ic = 0; ic < c_in; ic++) {
                sum += weight[oc * c_in + ic] * in[(size_t)ic * T + t];
            }
            out[(size_t)oc * T + t] = sum;
        }
    }
#endif

    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + (size_t)oc * T;
            for (int t = 0; t < T; t++) {
                row[t] += b;
            }
        }
    }
}

/* ReLU in-place */
static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/* Sigmoid in-place */
static void sigmoid_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

/* Tanh in-place */
static void tanh_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return NULL;
    return safetensors_get_f32(sf, t);
}

int tts_speaker_enc_init(tts_speaker_enc_ctx_t *ctx,
                         multi_safetensors_t *ms, int verbose) {
    memset(ctx, 0, sizeof(*ctx));
    char name[256];

    /* Check if speaker encoder weights exist */
    safetensors_file_t *sf = NULL;
    if (!multi_safetensors_find(ms, "speaker_encoder.blocks.0.conv.weight", &sf)) {
        if (verbose) printf("TTS speaker encoder: not found (non-Base model)\n");
        return -1;
    }

    /* Block 0: Conv1d(128->512, k=5) */
    ctx->block0_weight = load_f32(ms, "speaker_encoder.blocks.0.conv.weight");
    ctx->block0_bias = load_f32(ms, "speaker_encoder.blocks.0.conv.bias");
    if (!ctx->block0_weight || !ctx->block0_bias) {
        fprintf(stderr, "TTS speaker encoder: failed to load block 0\n");
        goto fail;
    }

    /* Blocks 1-3: SE-Res2Net */
    int dilations[3] = { 2, 3, 4 };
    for (int b = 0; b < SPKENC_NUM_SERES2NET; b++) {
        spkenc_seres2net_t *blk = &ctx->blocks[b];
        int bi = b + 1;  /* weight index: blocks.1, blocks.2, blocks.3 */
        blk->dilation = dilations[b];

        /* tdnn1 */
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn1.conv.weight", bi);
        blk->tdnn1_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn1.conv.bias", bi);
        blk->tdnn1_bias = load_f32(ms, name);

        /* Res2Net conv groups (7 convolutions for groups 1-7) */
        for (int g = 0; g < 7; g++) {
            snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.res2net_block.blocks.%d.conv.weight", bi, g);
            blk->res_weight[g] = load_f32(ms, name);
            snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.res2net_block.blocks.%d.conv.bias", bi, g);
            blk->res_bias[g] = load_f32(ms, name);
            if (!blk->res_weight[g] || !blk->res_bias[g]) {
                fprintf(stderr, "TTS speaker encoder: failed to load res2net conv %d in block %d\n", g, bi);
                goto fail;
            }
        }

        /* tdnn2 */
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn2.conv.weight", bi);
        blk->tdnn2_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn2.conv.bias", bi);
        blk->tdnn2_bias = load_f32(ms, name);

        /* SE */
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv1.weight", bi);
        blk->se_fc1_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv1.bias", bi);
        blk->se_fc1_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv2.weight", bi);
        blk->se_fc2_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv2.bias", bi);
        blk->se_fc2_bias = load_f32(ms, name);

        if (!blk->tdnn1_weight || !blk->tdnn2_weight ||
            !blk->se_fc1_weight || !blk->se_fc2_weight) {
            fprintf(stderr, "TTS speaker encoder: failed to load block %d\n", bi);
            goto fail;
        }
    }

    /* MFA: Conv1d(1536->1536, k=1) */
    ctx->mfa_weight = load_f32(ms, "speaker_encoder.mfa.conv.weight");
    ctx->mfa_bias = load_f32(ms, "speaker_encoder.mfa.conv.bias");
    if (!ctx->mfa_weight || !ctx->mfa_bias) {
        fprintf(stderr, "TTS speaker encoder: failed to load MFA\n");
        goto fail;
    }

    /* ASP */
    ctx->asp_conv1_weight = load_f32(ms, "speaker_encoder.asp.tdnn.conv.weight");
    ctx->asp_conv1_bias = load_f32(ms, "speaker_encoder.asp.tdnn.conv.bias");
    ctx->asp_conv2_weight = load_f32(ms, "speaker_encoder.asp.conv.weight");
    ctx->asp_conv2_bias = load_f32(ms, "speaker_encoder.asp.conv.bias");
    if (!ctx->asp_conv1_weight || !ctx->asp_conv2_weight) {
        fprintf(stderr, "TTS speaker encoder: failed to load ASP\n");
        goto fail;
    }

    /* FC */
    ctx->fc_weight = load_f32(ms, "speaker_encoder.fc.weight");
    ctx->fc_bias = load_f32(ms, "speaker_encoder.fc.bias");
    if (!ctx->fc_weight || !ctx->fc_bias) {
        fprintf(stderr, "TTS speaker encoder: failed to load FC\n");
        goto fail;
    }

    ctx->loaded = 1;
    if (verbose) printf("TTS speaker encoder: loaded (76 tensors, ECAPA-TDNN)\n");
    return 0;

fail:
    tts_speaker_enc_free(ctx);
    return -1;
}

void tts_speaker_enc_free(tts_speaker_enc_ctx_t *ctx) {
    free(ctx->block0_weight);
    free(ctx->block0_bias);

    for (int b = 0; b < SPKENC_NUM_SERES2NET; b++) {
        spkenc_seres2net_t *blk = &ctx->blocks[b];
        free(blk->tdnn1_weight);
        free(blk->tdnn1_bias);
        for (int g = 0; g < 7; g++) {
            free(blk->res_weight[g]);
            free(blk->res_bias[g]);
        }
        free(blk->tdnn2_weight);
        free(blk->tdnn2_bias);
        free(blk->se_fc1_weight);
        free(blk->se_fc1_bias);
        free(blk->se_fc2_weight);
        free(blk->se_fc2_bias);
    }

    free(ctx->mfa_weight);
    free(ctx->mfa_bias);
    free(ctx->asp_conv1_weight);
    free(ctx->asp_conv1_bias);
    free(ctx->asp_conv2_weight);
    free(ctx->asp_conv2_bias);
    free(ctx->fc_weight);
    free(ctx->fc_bias);

    memset(ctx, 0, sizeof(*ctx));
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* SE-Res2Net block forward pass.
 * Input/output: [512, T]. scratch must be large enough for all intermediates. */
static void seres2net_forward(spkenc_seres2net_t *blk,
                              float *x, int T, float *scratch) {
    int C = SPKENC_CHANNELS;  /* 512 */
    int S = SPKENC_SCALE;     /* 8 */
    int G = C / S;            /* 64 */

    /* Save input for residual */
    float *residual = scratch;
    memcpy(residual, x, (size_t)C * T * sizeof(float));
    float *work = residual + (size_t)C * T;

    /* tdnn1: Conv1d(512->512, k=1) + ReLU */
    float *tdnn1_out = work;
    conv1d_k1(tdnn1_out, x, blk->tdnn1_weight, blk->tdnn1_bias, C, C, T);
    relu_inplace(tdnn1_out, C * T);
    work += (size_t)C * T;

    /* Res2Net: split into 8 groups of 64, process groups */
    /* Group 0: passthrough */
    /* Groups 1-7: x[i] = conv(x[i] + prev_out) + ReLU, where prev_out starts as 0 */
    float *prev_out = work;
    memset(prev_out, 0, (size_t)G * T * sizeof(float));
    work += (size_t)G * T;

    float *conv_scratch = work;
    /* scratch for conv1d_same: need c_in * T_padded + c_in * kernel * T */
    /* For k=3, dil=D, pad=D, T_padded = T + 2*D */
    /* Need at least G * (T + 2*blk->dilation) + G * 3 * T */

    /* Output groups go back into tdnn1_out (overwrite in-place) */
    float *group_in = work + (size_t)G * (T + 2 * blk->dilation) + (size_t)G * 3 * T;
    for (int g = 1; g < S; g++) {
        float *src = tdnn1_out + (size_t)g * G * T;

        /* group_in = x[g] + prev_out */
        memcpy(group_in, src, (size_t)G * T * sizeof(float));
        for (int i = 0; i < G * T; i++) {
            group_in[i] += prev_out[i];
        }

        /* Conv1d(64->64, k=3, dilation=D, same_pad) + ReLU */
        conv1d_same(src, group_in,
                    blk->res_weight[g - 1], blk->res_bias[g - 1],
                    G, G, T, 3, blk->dilation, conv_scratch);
        relu_inplace(src, G * T);

        /* prev_out = this group's output */
        memcpy(prev_out, src, (size_t)G * T * sizeof(float));
    }

    /* tdnn2: Conv1d(512->512, k=1) + ReLU */
    float *tdnn2_out = x;  /* write directly to output */
    conv1d_k1(tdnn2_out, tdnn1_out, blk->tdnn2_weight, blk->tdnn2_bias, C, C, T);
    relu_inplace(tdnn2_out, C * T);

    /* SE: Squeeze-and-Excitation */
    {
        int B = SPKENC_SE_BOTTLENECK;  /* 128 */

        /* Global average pooling: mean over T for each channel */
        float se_pool[SPKENC_CHANNELS];
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *row = tdnn2_out + (size_t)c * T;
            for (int t = 0; t < T; t++) sum += row[t];
            se_pool[c] = sum / T;
        }

        /* FC1: [512] -> [128] + ReLU */
        float se_hidden[SPKENC_SE_BOTTLENECK];
        for (int i = 0; i < B; i++) {
            float sum = blk->se_fc1_bias[i];
            for (int j = 0; j < C; j++) {
                sum += blk->se_fc1_weight[i * C + j] * se_pool[j];
            }
            se_hidden[i] = sum > 0.0f ? sum : 0.0f;  /* ReLU */
        }

        /* FC2: [128] -> [512] + sigmoid */
        float se_scale[SPKENC_CHANNELS];
        for (int i = 0; i < C; i++) {
            float sum = blk->se_fc2_bias[i];
            for (int j = 0; j < B; j++) {
                sum += blk->se_fc2_weight[i * B + j] * se_hidden[j];
            }
            se_scale[i] = 1.0f / (1.0f + expf(-sum));  /* sigmoid */
        }

        /* Scale channels */
        for (int c = 0; c < C; c++) {
            float s = se_scale[c];
            float *row = tdnn2_out + (size_t)c * T;
            for (int t = 0; t < T; t++) row[t] *= s;
        }
    }

    /* Residual connection */
    for (int i = 0; i < C * T; i++) {
        x[i] += residual[i];
    }
}

int tts_speaker_enc_forward(tts_speaker_enc_ctx_t *ctx,
                            const float *mel, int n_frames,
                            float *out_embedding) {
    if (!ctx->loaded) return -1;

    int T = n_frames;
    int C = SPKENC_CHANNELS;  /* 512 */

    /* Compute scratch size needed:
     * - block0 conv scratch: 128 * (T + 4) + 128 * 5 * T
     * - seres2net scratch: C * T (residual) + C * T (tdnn1_out) +
     *   G * T (prev_out) + conv scratch + G * T (group_in)
     * - 3 block outputs for MFA: 3 * C * T
     * - mfa_out: 1536 * T
     * - asp scratch: 4608 * T + 128 * T + 1536 * T
     * Use a generous estimate. */
    size_t scratch_size = (size_t)(
        /* General buffers */
        (size_t)C * T * 6 +     /* various intermediate buffers */
        128 * (T + 10) +        /* block0 padding */
        128 * 5 * T +           /* block0 im2col */
        64 * (T + 10) +         /* res2net padding */
        64 * 3 * T +            /* res2net im2col */
        64 * T +                /* group_in */
        3 * C * T +             /* MFA concat */
        1536 * T * 2 +          /* MFA output + ASP hidden */
        4608 * T +              /* ASP concat */
        128 * T +               /* ASP conv1 */
        1536 * T +              /* ASP conv2 */
        4096                    /* padding */
    );

    float *scratch = (float *)malloc(scratch_size * sizeof(float));
    float *block_out = (float *)malloc((size_t)C * T * sizeof(float));
    float *conv_scratch = scratch;

    if (!scratch || !block_out) {
        free(scratch);
        free(block_out);
        return -1;
    }

    /* Block 0: Conv1d(128->512, k=5, same_pad=2) + ReLU */
    conv1d_same(block_out, mel,
                ctx->block0_weight, ctx->block0_bias,
                128, C, T, 5, 1, conv_scratch);
    relu_inplace(block_out, C * T);

    /* Save block outputs for MFA */
    float *mfa_inputs[3];  /* pointers to block 1, 2, 3 outputs */

    /* Blocks 1-3: SE-Res2Net */
    for (int b = 0; b < SPKENC_NUM_SERES2NET; b++) {
        seres2net_forward(&ctx->blocks[b], block_out, T, scratch);
        /* Save a copy for MFA */
        mfa_inputs[b] = (float *)malloc((size_t)C * T * sizeof(float));
        if (!mfa_inputs[b]) {
            for (int j = 0; j < b; j++) free(mfa_inputs[j]);
            free(scratch); free(block_out);
            return -1;
        }
        memcpy(mfa_inputs[b], block_out, (size_t)C * T * sizeof(float));
    }

    /* MFA: concat(block1, block2, block3) -> [1536, T] */
    int mfa_ch = 3 * C;  /* 1536 */
    float *mfa_concat = (float *)malloc((size_t)mfa_ch * T * sizeof(float));
    if (!mfa_concat) {
        for (int b = 0; b < 3; b++) free(mfa_inputs[b]);
        free(scratch); free(block_out);
        return -1;
    }

    for (int b = 0; b < 3; b++) {
        memcpy(mfa_concat + (size_t)b * C * T, mfa_inputs[b],
               (size_t)C * T * sizeof(float));
        free(mfa_inputs[b]);
    }

    /* MFA conv: Conv1d(1536->1536, k=1) + ReLU */
    float *mfa_out = block_out;  /* reuse, need at least 1536 * T but C=512 < 1536 */
    free(block_out);
    mfa_out = (float *)malloc((size_t)mfa_ch * T * sizeof(float));
    if (!mfa_out) { free(mfa_concat); free(scratch); return -1; }

    conv1d_k1(mfa_out, mfa_concat, ctx->mfa_weight, ctx->mfa_bias,
              mfa_ch, mfa_ch, T);
    relu_inplace(mfa_out, mfa_ch * T);

    /* ASP: Attentive Statistics Pooling */
    /* hidden = mfa_out [1536, T] */
    /* Compute mean and std across T for each channel */
    float *mean_vec = (float *)calloc(mfa_ch, sizeof(float));
    float *std_vec = (float *)calloc(mfa_ch, sizeof(float));
    if (!mean_vec || !std_vec) {
        free(mean_vec); free(std_vec);
        free(mfa_out); free(mfa_concat); free(scratch);
        return -1;
    }

    for (int c = 0; c < mfa_ch; c++) {
        float sum = 0.0f;
        const float *row = mfa_out + (size_t)c * T;
        for (int t = 0; t < T; t++) sum += row[t];
        mean_vec[c] = sum / T;
    }
    for (int c = 0; c < mfa_ch; c++) {
        float m = mean_vec[c];
        float var = 0.0f;
        const float *row = mfa_out + (size_t)c * T;
        for (int t = 0; t < T; t++) {
            float d = row[t] - m;
            var += d * d;
        }
        std_vec[c] = sqrtf(var / T + 1e-5f);
    }

    /* ASP input: concat(hidden, mean_broadcast, std_broadcast) -> [4608, T] */
    int asp_in_ch = 3 * mfa_ch;  /* 4608 */
    float *asp_input = (float *)malloc((size_t)asp_in_ch * T * sizeof(float));
    if (!asp_input) {
        free(mean_vec); free(std_vec);
        free(mfa_out); free(mfa_concat); free(scratch);
        return -1;
    }

    /* Copy hidden */
    memcpy(asp_input, mfa_out, (size_t)mfa_ch * T * sizeof(float));
    /* Broadcast mean */
    for (int c = 0; c < mfa_ch; c++) {
        float *row = asp_input + (size_t)(mfa_ch + c) * T;
        for (int t = 0; t < T; t++) row[t] = mean_vec[c];
    }
    /* Broadcast std */
    for (int c = 0; c < mfa_ch; c++) {
        float *row = asp_input + (size_t)(2 * mfa_ch + c) * T;
        for (int t = 0; t < T; t++) row[t] = std_vec[c];
    }

    free(mean_vec);
    free(std_vec);

    /* ASP conv1: Conv1d(4608->128, k=1) + ReLU + Tanh */
    float *asp_h = (float *)malloc(128 * T * sizeof(float));
    if (!asp_h) {
        free(asp_input); free(mfa_out); free(mfa_concat); free(scratch);
        return -1;
    }
    conv1d_k1(asp_h, asp_input, ctx->asp_conv1_weight, ctx->asp_conv1_bias,
              asp_in_ch, 128, T);
    relu_inplace(asp_h, 128 * T);
    tanh_inplace(asp_h, 128 * T);

    free(asp_input);

    /* ASP conv2: Conv1d(128->1536, k=1) */
    float *asp_attn = (float *)malloc((size_t)mfa_ch * T * sizeof(float));
    if (!asp_attn) {
        free(asp_h); free(mfa_out); free(mfa_concat); free(scratch);
        return -1;
    }
    conv1d_k1(asp_attn, asp_h, ctx->asp_conv2_weight, ctx->asp_conv2_bias,
              128, mfa_ch, T);
    free(asp_h);

    /* Softmax over T for each channel */
    for (int c = 0; c < mfa_ch; c++) {
        float *row = asp_attn + (size_t)c * T;
        float max_val = row[0];
        for (int t = 1; t < T; t++) {
            if (row[t] > max_val) max_val = row[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            row[t] = expf(row[t] - max_val);
            sum += row[t];
        }
        for (int t = 0; t < T; t++) {
            row[t] /= sum;
        }
    }

    /* Weighted mean and weighted std */
    float *w_mean = (float *)calloc(mfa_ch, sizeof(float));
    float *w_std = (float *)calloc(mfa_ch, sizeof(float));
    if (!w_mean || !w_std) {
        free(w_mean); free(w_std);
        free(asp_attn); free(mfa_out); free(mfa_concat); free(scratch);
        return -1;
    }

    for (int c = 0; c < mfa_ch; c++) {
        const float *attn_row = asp_attn + (size_t)c * T;
        const float *hidden_row = mfa_out + (size_t)c * T;
        float wm = 0.0f;
        for (int t = 0; t < T; t++) {
            wm += attn_row[t] * hidden_row[t];
        }
        w_mean[c] = wm;

        float ws2 = 0.0f;
        for (int t = 0; t < T; t++) {
            ws2 += attn_row[t] * hidden_row[t] * hidden_row[t];
        }
        float var = ws2 - wm * wm;
        if (var < 1e-10f) var = 1e-10f;
        w_std[c] = sqrtf(var);
    }

    free(asp_attn);
    free(mfa_out);
    free(mfa_concat);

    /* FC: concat(w_mean, w_std) -> [3072] -> Conv1d(3072->1024, k=1) -> [1024] */
    float *fc_input = (float *)malloc(2 * mfa_ch * sizeof(float));
    if (!fc_input) {
        free(w_mean); free(w_std); free(scratch);
        return -1;
    }
    memcpy(fc_input, w_mean, (size_t)mfa_ch * sizeof(float));
    memcpy(fc_input + mfa_ch, w_std, (size_t)mfa_ch * sizeof(float));
    free(w_mean);
    free(w_std);

    /* FC projection: [3072] -> [1024] (T=1 since pooled) */
    conv1d_k1(out_embedding, fc_input, ctx->fc_weight, ctx->fc_bias,
              2 * mfa_ch, SPKENC_EMBED_DIM, 1);

    free(fc_input);
    free(scratch);
    return 0;
}
