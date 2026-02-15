/*
 * tts_speaker_enc.h - ECAPA-TDNN speaker encoder for Qwen3-TTS
 *
 * Loads speaker_encoder.* weights from the Base model's model.safetensors
 * and computes 1024-dim speaker embeddings from 128-band mel spectrograms.
 *
 * Architecture:
 *   Block 0:  Conv1d(128->512, k=5, d=1) + ReLU
 *   Block 1-3: SE-Res2Net (scale=8, dilations 2/3/4)
 *   MFA:       concat(b1,b2,b3) -> Conv1d(1536->1536, k=1) + ReLU
 *   ASP:       Attentive Statistics Pooling -> [3072]
 *   FC:        Conv1d(3072->1024, k=1) -> [1024] speaker embedding
 */

#ifndef LOCAL_AI_TTS_SPEAKER_ENC_H
#define LOCAL_AI_TTS_SPEAKER_ENC_H

#include "qwen_asr_safetensors.h"

#define SPKENC_CHANNELS     512
#define SPKENC_SCALE        8     /* Res2Net scale (groups) */
#define SPKENC_SE_BOTTLENECK 128
#define SPKENC_EMBED_DIM    1024
#define SPKENC_NUM_SERES2NET 3    /* blocks 1-3 */

/* SE-Res2Net block weights */
typedef struct {
    /* tdnn1: Conv1d(512->512, k=1) + ReLU */
    float *tdnn1_weight;    /* [512, 512] */
    float *tdnn1_bias;      /* [512] */

    /* Res2Net: 7 Conv1d(64->64, k=3, dilation=D) per scale group */
    float *res_weight[7];   /* [64, 64, 3] each */
    float *res_bias[7];     /* [64] each */

    /* tdnn2: Conv1d(512->512, k=1) + ReLU */
    float *tdnn2_weight;    /* [512, 512] */
    float *tdnn2_bias;      /* [512] */

    /* SE: global_avg_pool -> Conv1d(512->128,k=1) -> ReLU -> Conv1d(128->512,k=1) -> sigmoid */
    float *se_fc1_weight;   /* [128, 512] */
    float *se_fc1_bias;     /* [128] */
    float *se_fc2_weight;   /* [512, 128] */
    float *se_fc2_bias;     /* [512] */

    int dilation;           /* 2, 3, or 4 */
} spkenc_seres2net_t;

/* Speaker encoder context */
typedef struct {
    /* Block 0: Conv1d(128->512, k=5, same_pad=2) + ReLU */
    float *block0_weight;   /* [512, 128, 5] */
    float *block0_bias;     /* [512] */

    /* Blocks 1-3: SE-Res2Net */
    spkenc_seres2net_t blocks[SPKENC_NUM_SERES2NET];

    /* MFA: Conv1d(1536->1536, k=1) + ReLU */
    float *mfa_weight;      /* [1536, 1536] */
    float *mfa_bias;        /* [1536] */

    /* ASP attention */
    float *asp_conv1_weight; /* [128, 4608] */
    float *asp_conv1_bias;   /* [128] */
    float *asp_conv2_weight; /* [1536, 128] */
    float *asp_conv2_bias;   /* [1536] */

    /* FC: Conv1d(3072->1024, k=1) */
    float *fc_weight;       /* [1024, 3072] */
    float *fc_bias;         /* [1024] */

    int loaded;             /* 1 if weights loaded successfully */
} tts_speaker_enc_ctx_t;

/* Initialize speaker encoder from model safetensors.
 * Returns 0 on success, -1 if speaker_encoder.* tensors not found
 * (non-Base model -- not an error, just means no voice cloning). */
int tts_speaker_enc_init(tts_speaker_enc_ctx_t *ctx,
                         multi_safetensors_t *ms, int verbose);

/* Free all speaker encoder weights. */
void tts_speaker_enc_free(tts_speaker_enc_ctx_t *ctx);

/* Forward pass: mel [128, n_frames] -> out_embedding [1024].
 * Returns 0 on success. */
int tts_speaker_enc_forward(tts_speaker_enc_ctx_t *ctx,
                            const float *mel, int n_frames,
                            float *out_embedding);

#endif /* LOCAL_AI_TTS_SPEAKER_ENC_H */
