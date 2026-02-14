/*
 * tts_vocoder_ops.c - Vocoder operations for native Qwen3-TTS vocoder
 *
 * Conv1d (causal, im2col+SGEMM), ConvTranspose1d, SnakeBeta activation,
 * ConvNeXt blocks, BigVGAN decoder blocks, LayerNorm, GELU.
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
 * Conv1d (causal) -- im2col + SGEMM
 * ======================================================================== */

/*
 * im2col for 1D: unroll input patches for GEMM-based conv1d.
 * Input: [C_in, T_padded] (already left-padded)
 * Output columns: [C_in * kernel, T_out]
 */
static void im2col_1d(const float *in, float *cols,
                       int c_in, int T_padded, int kernel, int dilation,
                       int T_out) {
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kernel; ki++) {
            int col_row = ic * kernel + ki;
            float *col_ptr = cols + (size_t)col_row * T_out;
            const float *in_ch = in + (size_t)ic * T_padded;
            for (int t = 0; t < T_out; t++) {
                col_ptr[t] = in_ch[t * 1 + ki * dilation]; /* stride=1 always */
            }
        }
    }
}

void voc_conv1d_causal(float *out, const float *in,
                       const float *weight, const float *bias,
                       int c_in, int c_out, int T, int kernel, int dilation,
                       int groups, float *scratch) {
    int pad = dilation * (kernel - 1);  /* causal: left-pad only */
    int T_padded = T + pad;

    if (groups > 1 && groups == c_in && groups == c_out) {
        /* Depthwise convolution: direct loop (small kernel, typically k=7) */
        /* Pad input per-channel and convolve */
        for (int ch = 0; ch < c_in; ch++) {
            /* Build padded channel in scratch */
            float *padded = scratch + (size_t)ch * T_padded;
            memset(padded, 0, (size_t)pad * sizeof(float));
            memcpy(padded + pad, in + (size_t)ch * T, (size_t)T * sizeof(float));
        }
        for (int ch = 0; ch < c_in; ch++) {
            const float *padded = scratch + (size_t)ch * T_padded;
            const float *w = weight + (size_t)ch * kernel; /* [1, kernel] per group */
            float *o = out + (size_t)ch * T;
            float b = bias ? bias[ch] : 0.0f;
            for (int t = 0; t < T; t++) {
                float sum = b;
                for (int ki = 0; ki < kernel; ki++) {
                    sum += w[ki] * padded[t + ki * dilation];
                }
                o[t] = sum;
            }
        }
        return;
    }

    /* Standard (non-grouped) conv1d via im2col + SGEMM */
    /* Allocate padded input: [c_in, T_padded] */
    float *padded = scratch;
    for (int ch = 0; ch < c_in; ch++) {
        float *dst = padded + (size_t)ch * T_padded;
        memset(dst, 0, (size_t)pad * sizeof(float));
        memcpy(dst + pad, in + (size_t)ch * T, (size_t)T * sizeof(float));
    }

    /* im2col: [c_in * kernel, T] */
    int patch_size = c_in * kernel;
    float *cols = padded + (size_t)c_in * T_padded; /* use scratch after padded */
    im2col_1d(padded, cols, c_in, T_padded, kernel, dilation, T);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, T] = out[c_out, T] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, T, patch_size,
                1.0f, weight, patch_size, cols, T,
                0.0f, out, T);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int p = 0; p < patch_size; p++) {
                sum += weight[oc * patch_size + p] * cols[(size_t)p * T + t];
            }
            out[oc * T + t] = sum;
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

/* ========================================================================
 * ConvTranspose1d (causal)
 * ======================================================================== */

void voc_conv_transpose1d(float *out, const float *in,
                          const float *weight, const float *bias,
                          int c_in, int c_out, int T, int kernel, int stride) {
    /* Raw output: (T-1)*stride + kernel positions.
     * Causal trim: remove (kernel-stride) from left AND right.
     * Final output length: (T+1)*stride - kernel.
     * For BigVGAN (kernel=2*stride): T_out = (T-1)*stride.
     * For ConvNeXt upsample (kernel=stride): T_out = T*stride (no trim). */
    int trim = kernel - stride;
    int T_out = (T + 1) * stride - kernel;

    /* Zero output (accumulate via scatter-add) */
    memset(out, 0, (size_t)c_out * T_out * sizeof(float));

    /* weight layout: [c_in, c_out, kernel] */
    for (int ic = 0; ic < c_in; ic++) {
        for (int oc = 0; oc < c_out; oc++) {
            const float *w = weight + ((size_t)ic * c_out + oc) * kernel;
            const float *x = in + (size_t)ic * T;
            float *y = out + (size_t)oc * T_out;
            for (int t = 0; t < T; t++) {
                float xval = x[t];
                int out_start = t * stride - trim;
                for (int ki = 0; ki < kernel; ki++) {
                    int oi = out_start + ki;
                    if (oi >= 0 && oi < T_out) {
                        y[oi] += xval * w[ki];
                    }
                }
            }
        }
    }

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + (size_t)oc * T_out;
            for (int t = 0; t < T_out; t++) {
                row[t] += b;
            }
        }
    }
}

/* ========================================================================
 * SnakeBeta activation
 * ======================================================================== */

void voc_snake_beta(float *x, const float *exp_alpha, const float *inv_exp_beta,
                    int channels, int T) {
    /* x += inv_exp_beta * sin^2(exp_alpha * x) */
    for (int ch = 0; ch < channels; ch++) {
        float ea = exp_alpha[ch];
        float ieb = inv_exp_beta[ch];
        float *row = x + (size_t)ch * T;
        for (int t = 0; t < T; t++) {
            float s = sinf(ea * row[t]);
            row[t] += ieb * s * s;
        }
    }
}

/* ========================================================================
 * LayerNorm over channels (for ConvNeXt)
 * ======================================================================== */

void voc_layer_norm_channels(float *x, const float *weight, const float *bias,
                             int channels, int T, float eps) {
    /* x is [C, T]. For each time step t, normalize x[:, t] across C.
     * This is equivalent to transposing to [T, C], applying LN, transposing back. */
    for (int t = 0; t < T; t++) {
        /* Compute mean */
        float mean = 0.0f;
        for (int c = 0; c < channels; c++) {
            mean += x[(size_t)c * T + t];
        }
        mean /= channels;

        /* Compute variance */
        float var = 0.0f;
        for (int c = 0; c < channels; c++) {
            float d = x[(size_t)c * T + t] - mean;
            var += d * d;
        }
        var /= channels;

        /* Normalize */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < channels; c++) {
            float val = (x[(size_t)c * T + t] - mean) * inv_std;
            x[(size_t)c * T + t] = val * weight[c] + bias[c];
        }
    }
}

/* ========================================================================
 * GELU (exact)
 * ======================================================================== */

void voc_gelu(float *x, int n) {
    const float sqrt2_inv = 0.7071067811865475f; /* 1/sqrt(2) */
    for (int i = 0; i < n; i++) {
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * sqrt2_inv));
    }
}
