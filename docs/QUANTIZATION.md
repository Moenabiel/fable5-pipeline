# Quantization Guide

## Overview

This guide covers converting HuggingFace models to CPU-runnable quantized formats (GGUF for llama.cpp, GPTQ for transformers).

## Option 1: llama.cpp GGUF (Recommended for CPU)

### Prerequisites
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)
pip install gguf sentencepiece
```

### Step 1: Download Model
```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b-instruct
```

### Step 2: Convert to GGUF
```bash
cd llama.cpp
python convert_hf_to_gguf.py ../models/mistral-7b-instruct --outfile ../models/mistral-7b-instruct-f16.bin --outtype f16
```

### Step 3: Quantize
```bash
# Q4_0 (4-bit, smallest, fastest on CPU)
./quantize ../models/mistral-7b-instruct-f16.bin ../models/mistral-7b-instruct-q4_0.bin q4_0

# Q4_K_M (4-bit, medium quality — recommended balance)
./quantize ../models/mistral-7b-instruct-f16.bin ../models/mistral-7b-instruct-q4_k_m.bin q4_k_m

# Q5_K_M (5-bit, better quality, larger)
./quantize ../models/mistral-7b-instruct-f16.bin ../models/mistral-7b-instruct-q5_k_m.bin q5_k_m
```

### Step 4: Run
```bash
./main -m ../models/mistral-7b-instruct-q4_0.bin -p "Hello, how are you?" -n 256 --temp 0.2
```

## Option 2: GPTQ (For transformers with GPU)

### Prerequisites
```bash
pip install auto-gptq optimum
```

### Convert
```bash
python -m auto_gptq.modeling._utils \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --bits 4 \
    --group_size 128 \
    --output_dir models/mistral-7b-instruct-gptq-4bit
```

### Run
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("models/mistral-7b-instruct-gptq-4bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("models/mistral-7b-instruct-gptq-4bit")
```

## Option 3: bitsandbytes 4-bit (Colab GPU, no conversion needed)

The `hf_bnb_runner` backend handles this automatically:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantization_config=bnb_config, device_map="auto")
```

## Size Estimates

| Model | FP16 | Q4_0 | Q4_K_M | Q5_K_M |
|-------|------|------|--------|--------|
| Mistral-7B | ~14GB | ~4.1GB | ~4.5GB | ~5.3GB |
| Llama-2-7B | ~14GB | ~4.1GB | ~4.5GB | ~5.3GB |
| Qwen2-7B | ~14GB | ~4.1GB | ~4.5GB | ~5.3GB |
| TinyLlama-1.1B | ~2.2GB | ~0.7GB | ~0.8GB | ~0.9GB |

## Performance Notes

- **Q4_0**: Fastest on CPU, lowest quality. Good for testing.
- **Q4_K_M**: Best balance of speed and quality. Recommended for production.
- **Q5_K_M**: Higher quality, slightly slower. Use when quality matters more.
- **CPU inference**: ~2-5 tokens/sec on modern CPU for 7B Q4_0.
- **GPU inference**: ~30-50 tokens/sec on free Colab T4 for 7B 4-bit.
