/*
 * tts_sampling.c - Token sampling for TTS autoregressive decode
 *
 * Temperature scaling, top-k filtering, and repetition penalty.
 */

#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>

/* Apply repetition penalty to logits for previously generated tokens.
 * Tokens in history[] get their logits divided (if positive) or
 * multiplied (if negative) by the penalty factor.
 * Each unique token is penalized exactly once (matching reference
 * implementations which use set-based deduplication). */
void tts_apply_repetition_penalty(float *logits, int vocab_size,
                                  const int *history, int history_len,
                                  float penalty) {
    if (penalty <= 1.0f) return;
    uint8_t *seen = (uint8_t *)calloc((size_t)vocab_size, 1);
    if (!seen) return;
    for (int i = 0; i < history_len; i++) {
        int tok = history[i];
        if (tok < 0 || tok >= vocab_size) continue;
        if (seen[tok]) continue;
        seen[tok] = 1;
        if (logits[tok] > 0.0f) {
            logits[tok] /= penalty;
        } else {
            logits[tok] *= penalty;
        }
    }
    free(seen);
}

/* Sample a token from logits with temperature scaling and top-k filtering.
 * Returns the selected token index. */
int tts_sample_topk(float *logits, int vocab_size, float temperature, int top_k) {
    /* Find top_k indices */
    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;

    /* Simple partial sort: find top_k largest values */
    int *indices = (int *)malloc((size_t)top_k * sizeof(int));
    float *values = (float *)malloc((size_t)top_k * sizeof(float));
    if (!indices || !values) {
        free(indices);
        free(values);
        return 0;
    }

    int count = 0;
    float min_val = -FLT_MAX;
    int min_idx = 0;

    for (int i = 0; i < vocab_size; i++) {
        if (count < top_k) {
            indices[count] = i;
            values[count] = logits[i];
            if (count == 0 || logits[i] < min_val) {
                min_val = logits[i];
                min_idx = count;
            }
            count++;
        } else if (logits[i] > min_val) {
            indices[min_idx] = i;
            values[min_idx] = logits[i];
            /* Find new minimum */
            min_val = values[0];
            min_idx = 0;
            for (int j = 1; j < top_k; j++) {
                if (values[j] < min_val) {
                    min_val = values[j];
                    min_idx = j;
                }
            }
        }
    }

    /* Apply temperature and compute softmax */
    float max_logit = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] > max_logit) max_logit = values[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        values[i] = expf((values[i] - max_logit) / temperature);
        sum += values[i];
    }

    /* Sample from distribution */
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cumulative = 0.0f;
    int selected = indices[0];
    for (int i = 0; i < count; i++) {
        cumulative += values[i];
        if (cumulative >= r) {
            selected = indices[i];
            break;
        }
    }

    free(indices);
    free(values);
    return selected;
}
