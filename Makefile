# local-ai-server — Linux/macOS Makefile
# Mirrors build.bat compilation tiers:
#   qwen-asr + vocoder: optimized (-O2 -march=native -ffast-math)
#   server sources:     debug (-g -DDEBUG)

CC = gcc
CFLAGS_OPT  = -Wall -Wextra -O2 -march=native -ffast-math -DNDEBUG
CFLAGS_SRV  = -Wall -Wextra -g -DDEBUG
LDFLAGS     = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Directories
QWEN_DIR  = qwen-asr
SRC_DIR   = src
BUILD_DIR = build
BIN_DIR   = bin

# Include paths
INCLUDES = -I$(QWEN_DIR) -I$(SRC_DIR)

# ---- Source files ----

# qwen-asr (excluding main.c and gpu.c -- no CUDA in this build)
QWEN_SRCS = \
	$(QWEN_DIR)/qwen_asr.c \
	$(QWEN_DIR)/qwen_asr_audio.c \
	$(QWEN_DIR)/qwen_asr_decoder.c \
	$(QWEN_DIR)/qwen_asr_encoder.c \
	$(QWEN_DIR)/qwen_asr_kernels.c \
	$(QWEN_DIR)/qwen_asr_kernels_avx.c \
	$(QWEN_DIR)/qwen_asr_kernels_generic.c \
	$(QWEN_DIR)/qwen_asr_kernels_neon.c \
	$(QWEN_DIR)/qwen_asr_safetensors.c \
	$(QWEN_DIR)/qwen_asr_tokenizer.c

# Vocoder (optimized)
VOC_SRCS = \
	$(SRC_DIR)/tts_vocoder.c \
	$(SRC_DIR)/tts_vocoder_ops.c \
	$(SRC_DIR)/tts_vocoder_xfmr.c \
	$(SRC_DIR)/tts_mel.c \
	$(SRC_DIR)/tts_speaker_enc.c

# Server (debug build)
SRV_SRCS = \
	$(SRC_DIR)/main.c \
	$(SRC_DIR)/http.c \
	$(SRC_DIR)/multipart.c \
	$(SRC_DIR)/handler_asr.c \
	$(SRC_DIR)/handler_tts.c \
	$(SRC_DIR)/json.c \
	$(SRC_DIR)/json_reader.c \
	$(SRC_DIR)/tts_pipeline.c \
	$(SRC_DIR)/tts_sampling.c \
	$(SRC_DIR)/tts_native.c \
	$(SRC_DIR)/tts_voice_presets.c

# Object files (all go into build/)
QWEN_OBJS = $(patsubst $(QWEN_DIR)/%.c,$(BUILD_DIR)/%.o,$(QWEN_SRCS))
VOC_OBJS  = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(VOC_SRCS))
SRV_OBJS  = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRV_SRCS))

TARGET = $(BIN_DIR)/local-ai-server

.PHONY: all help blas debug clean info

all: help

help:
	@echo "local-ai-server — Linux/macOS Build"
	@echo ""
	@echo "Targets:"
	@echo "  make blas   - Build with BLAS (Accelerate/OpenBLAS)"
	@echo "  make debug  - Debug build with AddressSanitizer"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make info   - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./bin/local-ai-server --help"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: BLAS_CFLAGS = -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: BLAS_LDFLAGS = -framework Accelerate
else
blas: BLAS_CFLAGS = -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: BLAS_LDFLAGS = -lopenblas
endif
blas:
	@$(MAKE) build_server BLAS_CFLAGS="$(BLAS_CFLAGS)" BLAS_LDFLAGS="$(BLAS_LDFLAGS)"

# Debug build with AddressSanitizer
debug: BLAS_CFLAGS =
debug: BLAS_LDFLAGS =
debug:
	@$(MAKE) build_server \
		CFLAGS_OPT="-Wall -Wextra -g -O0 -DDEBUG -fsanitize=address" \
		CFLAGS_SRV="-Wall -Wextra -g -O0 -DDEBUG -fsanitize=address" \
		BLAS_CFLAGS="" BLAS_LDFLAGS="" \
		EXTRA_LDFLAGS="-fsanitize=address"

# =============================================================================
# Build rules
# =============================================================================
build_server: $(TARGET)
	@echo ""
	@echo "Build complete: $(TARGET)"

$(TARGET): $(QWEN_OBJS) $(VOC_OBJS) $(SRV_OBJS) | $(BIN_DIR)
	$(CC) -o $@ $^ $(LDFLAGS) $(BLAS_LDFLAGS) $(EXTRA_LDFLAGS)

# Static pattern rules avoid ambiguity (qwen-asr/ and src/ both have main.c)

# qwen-asr sources: optimized
$(QWEN_OBJS): $(BUILD_DIR)/%.o: $(QWEN_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS_OPT) $(BLAS_CFLAGS) $(INCLUDES) -c -o $@ $<

# Vocoder sources: optimized
$(VOC_OBJS): $(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS_OPT) $(BLAS_CFLAGS) $(INCLUDES) -c -o $@ $<

# Server sources: debug
$(SRV_OBJS): $(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS_SRV) $(BLAS_CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
ifeq ($(UNAME_S),Darwin)
	@echo "BLAS:     Apple Accelerate"
else
	@echo "BLAS:     OpenBLAS"
endif
	@echo "CUDA:     disabled (Linux build)"
