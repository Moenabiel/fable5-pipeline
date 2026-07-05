#!/bin/bash
# Fable5 Pipeline — Local CPU Demo
# Requirements: quantized GGUF model, llama.cpp compiled

set -e

MODEL_PATH="${1:-models/mistral-7b-instruct-q4_0.bin}"
LLAMA_CPP="${2:-./llama.cpp/main}"
PROMPT="${3:-Explain quantum computing in simple terms}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo ""
    echo "Download and quantize a model first:"
    echo "  See docs/QUANTIZATION.md for instructions"
    echo ""
    echo "Quick start with TinyLlama (smaller, faster):"
    echo "  huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama"
    echo "  cd llama.cpp && python convert_hf_to_gguf.py ../models/tinyllama --outfile ../models/tinyllama-f16.bin --outtype f16"
    echo "  ./quantize ../models/tinyllama-f16.bin ../models/tinyllama-q4_0.bin q4_0"
    exit 1
fi

echo "Running Fable5 pipeline..."
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo ""

python -m src.orchestrator \
    --prompt "$PROMPT" \
    --backend llama_cpp \
    --model-path "$MODEL_PATH" \
    --llama-cpp-bin "$LLAMA_CPP" \
    --k 3 \
    --log-dir logs
