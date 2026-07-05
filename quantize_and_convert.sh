#!/bin/bash
# Quantization and conversion script for the Fable5 pipeline
# Converts HuggingFace models to CPU-runnable GGUF format

set -e

MODEL_ID="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
OUTPUT_DIR="models"
QUANT_TYPE="${2:-q4_k_m}"

echo "=== Fable5 Model Quantization ==="
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "Quantization: $QUANT_TYPE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make -j$(nproc)
    cd ..
fi

# Install dependencies
pip install gguf sentencepiece huggingface_hub

# Download model
echo "Downloading model..."
MODEL_NAME=$(echo "$MODEL_ID" | tr '/' '_')
huggingface-cli download "$MODEL_ID" --local-dir "$OUTPUT_DIR/$MODEL_NAME"

# Convert to GGUF
echo "Converting to GGUF..."
cd llama.cpp
python convert_hf_to_gguf.py "../$OUTPUT_DIR/$MODEL_NAME" --outfile "../$OUTPUT_DIR/${MODEL_NAME}-f16.bin" --outtype f16
cd ..

# Quantize
echo "Quantizing to $QUANT_TYPE..."
cd llama.cpp
./quantize "../$OUTPUT_DIR/${MODEL_NAME}-f16.bin" "../$OUTPUT_DIR/${MODEL_NAME}-${QUANT_TYPE}.bin" "$QUANT_TYPE"
cd ..

echo ""
echo "=== Done ==="
echo "Quantized model: $OUTPUT_DIR/${MODEL_NAME}-${QUANT_TYPE}.bin"
echo ""
echo "Run with:"
echo "  python -m src.orchestrator --prompt 'Your question' --backend llama_cpp --model-path $OUTPUT_DIR/${MODEL_NAME}-${QUANT_TYPE}.bin"
