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

    /* Standard (non-grouped) conv1d */
    int patch_size = c_in * kernel;

    /* Pad input: [c_in, T_padded] */
    float *padded = scratch;
    for (int ch = 0; ch < c_in; ch++) {
        float *dst = padded + (size_t)ch * T_padded;
        memset(dst, 0, (size_t)pad * sizeof(float));
        memcpy(dst + pad, in + (size_t)ch * T, (size_t)T * sizeof(float));
    }

#ifdef USE_BLAS
    /* Choose between im2col + single SGEMM vs implicit GEMM (no im2col).
     * Implicit GEMM avoids the large im2col buffer but uses kernel separate SGEMM
     * calls. It wins when the im2col buffer far exceeds L3 cache; loses when the
     * buffer is small (the single large-K SGEMM is more efficient). */
    size_t im2col_size = (size_t)c_in * kernel * T;
    if (im2col_size <= 10000000) {
        /* Small im2col: standard approach (fits in cache, single SGEMM is faster) */
        float *cols = padded + (size_t)c_in * T_padded;
        im2col_1d(padded, cols, c_in, T_padded, kernel, dilation, T);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    c_out, T, patch_size,
                    1.0f, weight, patch_size, cols, T,
                    0.0f, out, T);
    } else {
        /* Large im2col: implicit GEMM (one SGEMM per kernel tap, no buffer).
         * Eliminates c_in * kernel * T im2col buffer (up to 250+ MB). */
        float *w_slice = padded + (size_t)c_in * T_padded;

        for (int ki = 0; ki < kernel; ki++) {
            /* Extract weight slice: w_slice[oc][ic] = weight[oc*patch + ic*kernel + ki] */
            for (int oc = 0; oc < c_out; oc++) {
                const float *src = weight + (size_t)oc * patch_size + ki;
                float *dst = w_slice + (size_t)oc * c_in;
                for (int ic = 0; ic < c_in; ic++) {
                    dst[ic] = src[ic * kernel];
                }
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        c_out, T, c_in,
                        1.0f, w_slice, c_in,
                        padded + ki * dilation, T_padded,
                        ki == 0 ? 0.0f : 1.0f, out, T);
        }
    }
#else
    /* im2col + naive GEMM fallback (no BLAS) */
    float *cols = padded + (size_t)c_in * T_padded;
    im2col_1d(padded, cols, c_in, T_padded, kernel, dilation, T);

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
                          int c_in, int c_out, int T, int kernel, int stride,
                          float *scratch) {
    /* Raw output: (T-1)*stride + kernel positions.
     * Causal trim: remove (kernel-stride) from left AND right.
     * Final output length: (T+1)*stride - kernel.
     * For BigVGAN (kernel=2*stride): T_out = (T-1)*stride.
     * For ConvNeXt upsample (kernel=stride): T_out = T*stride (no trim). */
    int trim = kernel - stride;
    int T_out = (T + 1) * stride - kernel;

    /* Zero output (accumulate via scatter-add) */
    memset(out, 0, (size_t)c_out * T_out * sizeof(float));

#ifdef USE_BLAS
    /* GEMM approach: weight [c_in, c_out*kernel] transposed -> [c_out*kernel, c_in]
     * cols[c_out*kernel, T] = W^T @ X, then col2im scatter-add (no multiply). */
    int M = c_out * kernel;
    float *cols = scratch;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, T, c_in,
                1.0f, weight, M, in, T,
                0.0f, cols, T);

    /* col2im: scatter columns into output */
    for (int oc = 0; oc < c_out; oc++) {
        float *y = out + (size_t)oc * T_out;
        for (int ki = 0; ki < kernel; ki++) {
            const float *col_row = cols + (size_t)(oc * kernel + ki) * T;
            for (int t = 0; t < T; t++) {
                int oi = t * stride + ki - trim;
                if (oi >= 0 && oi < T_out) {
                    y[oi] += col_row[t];
                }
            }
        }
    }
#else
    /* Naive scatter-add fallback (no BLAS) */
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
#endif

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

/* ---- AVX2 fast sin approximation for SnakeBeta ---- */
#include <immintrin.h>

/*
 * Fast sin(x) using AVX2 with Cephes-style range reduction.
 * Accuracy: max error ~2e-7 over all float inputs.
 *
 * Method:
 *   1. Range reduce: n = round(x / pi), r = x - n * pi
 *   2. Polynomial: sin(r) ~ r * (1 + r^2*(c3 + r^2*(c5 + r^2*c7)))
 *   3. Negate if n is odd (sin has period 2*pi, sign flips every pi)
 */
static inline __m256 fast_sin_avx2(__m256 x) {
    const __m256 inv_pi = _mm256_set1_ps(0.3183098861837907f);
    const __m256 pi_hi  = _mm256_set1_ps(3.1415927410125732f);  /* high bits of pi */
    const __m256 pi_lo  = _mm256_set1_ps(-8.742278012618954e-8f); /* pi - pi_hi */

    /* Range reduce: n = round(x / pi) */
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, inv_pi),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* r = x - n*pi (Cody-Waite two-step for precision) */
    __m256 r = _mm256_fnmadd_ps(n, pi_hi, x);
    r = _mm256_fnmadd_ps(n, pi_lo, r);

    /* Sign: negate result if n is odd */
    __m256i ni = _mm256_cvtps_epi32(n);
    __m256i sign_mask = _mm256_slli_epi32(_mm256_and_si256(ni, _mm256_set1_epi32(1)), 31);

    /* Polynomial: sin(r) ~ r * (c1 + r^2*(c3 + r^2*(c5 + r^2*c7)))
     * Coefficients from minimax fit on [-pi/2, pi/2] */
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p = _mm256_set1_ps(-1.9515295891e-4f);              /* c7 */
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(8.3321608736e-3f));  /* c5 */
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.6666654611e-1f)); /* c3 */
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(1.0f));              /* c1 */
    __m256 result = _mm256_mul_ps(r, p);

    /* Apply sign flip */
    result = _mm256_xor_ps(result, _mm256_castsi256_ps(sign_mask));
    return result;
}

void voc_snake_beta(float *x, const float *exp_alpha, const float *inv_exp_beta,
                    int channels, int T) {
    /* x += inv_exp_beta * sin^2(exp_alpha * x) */
    int T8 = T & ~7;

    for (int ch = 0; ch < channels; ch++) {
        __m256 ea = _mm256_set1_ps(exp_alpha[ch]);
        __m256 ieb = _mm256_set1_ps(inv_exp_beta[ch]);
        float *row = x + (size_t)ch * T;

        int t = 0;
        for (; t < T8; t += 8) {
            __m256 v = _mm256_loadu_ps(row + t);
            __m256 s = fast_sin_avx2(_mm256_mul_ps(ea, v));
            __m256 s2 = _mm256_mul_ps(s, s);
            _mm256_storeu_ps(row + t, _mm256_fmadd_ps(ieb, s2, v));
        }
        /* Scalar tail */
        float ea_s = exp_alpha[ch];
        float ieb_s = inv_exp_beta[ch];
        for (; t < T; t++) {
            float s = sinf(ea_s * row[t]);
            row[t] += ieb_s * s * s;
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
