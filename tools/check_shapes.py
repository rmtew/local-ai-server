"""Check weight shapes for upsample and BigVGAN operations."""
import struct
import json

MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"

with open(MODEL, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_size))

# Print shapes for key weights
keys = [
    # Upsample transposed convs
    "decoder.upsample.0.0.conv.weight",
    "decoder.upsample.0.0.conv.bias",
    "decoder.upsample.1.0.conv.weight",
    # ConvNeXt dwconv
    "decoder.upsample.0.1.dwconv.conv.weight",
    "decoder.upsample.0.1.dwconv.conv.bias",
    # ConvNeXt pointwise
    "decoder.upsample.0.1.pwconv1.weight",
    "decoder.upsample.0.1.pwconv1.bias",
    "decoder.upsample.0.1.pwconv2.weight",
    "decoder.upsample.0.1.pwconv2.bias",
    "decoder.upsample.0.1.gamma",
    # BigVGAN init conv
    "decoder.decoder.0.conv.weight",
    "decoder.decoder.0.conv.bias",
    # BigVGAN block 0 snake
    "decoder.decoder.1.block.0.alpha",
    "decoder.decoder.1.block.0.beta",
    # BigVGAN block 0 transconv
    "decoder.decoder.1.block.1.conv.weight",
    "decoder.decoder.1.block.1.conv.bias",
    # BigVGAN block 0 resunit 0
    "decoder.decoder.1.block.2.act1.alpha",
    "decoder.decoder.1.block.2.conv1.conv.weight",
    "decoder.decoder.1.block.2.conv1.conv.bias",
    "decoder.decoder.1.block.2.act2.alpha",
    "decoder.decoder.1.block.2.conv2.conv.weight",
    "decoder.decoder.1.block.2.conv2.conv.bias",
    # BigVGAN block 3 transconv (rate=3)
    "decoder.decoder.4.block.1.conv.weight",
    "decoder.decoder.4.block.1.conv.bias",
    # Final
    "decoder.decoder.5.alpha",
    "decoder.decoder.6.conv.weight",
    "decoder.decoder.6.conv.bias",
]

for k in keys:
    if k in header:
        info = header[k]
        print(f"{k}: shape={info['shape']}, dtype={info['dtype']}")
    else:
        print(f"{k}: NOT FOUND")

# Also check the ConvNeXt - is pwconv1 a Linear or Conv1d weight?
print("\n--- Analysis ---")
pw1 = header.get("decoder.upsample.0.1.pwconv1.weight")
if pw1:
    shape = pw1["shape"]
    print(f"pwconv1 shape: {shape}")
    if len(shape) == 2:
        print(f"  -> Linear weight [out={shape[0]}, in={shape[1]}]")
    elif len(shape) == 3:
        print(f"  -> Conv1d weight [out={shape[0]}, in={shape[1]}, kernel={shape[2]}]")

# Check BigVGAN transconv weight layout
tc = header.get("decoder.decoder.1.block.1.conv.weight")
if tc:
    shape = tc["shape"]
    print(f"BigVGAN block0 tconv shape: {shape}")
    print(f"  -> ConvTranspose1d [in={shape[0]}, out={shape[1]}, kernel={shape[2]}]")
    print(f"  Expected: [1536, 768, 16] (in=1536, out=768, kernel=2*rate=16)")
