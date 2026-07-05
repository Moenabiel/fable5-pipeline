# Fable5-Pipeline: Open-Source Approximation of Claude Fable 5

A zero-dollar, fully open pipeline that approximates the behavior and output quality of Anthropic's "Claude Fable 5" at the behavior level — structured plans, step-by-step reasoning, evidence checks, and polished final answers — using only openly available, legally usable resources.

## Quick Start

### Option A: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_starter.ipynb)

### Option B: Local CPU (Quantized Model)
```bash
# 1. Clone & install
git clone https://github.com/yourname/fable5-pipeline.git
cd fable5-pipeline
pip install -r requirements.txt

# 2. Download a quantized model (or use llama.cpp to convert)
# See docs/QUANTIZATION.md for exact commands

# 3. Run the orchestrator
python -m src.orchestrator --prompt "Explain quantum computing in simple terms" --backend llama_cpp --model-path models/mistral-7b-instruct-q4_0.bin
```

### Option C: Colab GPU (4-bit quantized)
```bash
pip install transformers accelerate bitsandbytes peft
python -m src.orchestrator --prompt "Write a business plan for a coffee shop" --backend hf_bnb --model-id mistralai/Mistral-7B-Instruct-v0.3
```

## Architecture

```
User Goal
    │
    ▼
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│ Planner  │────▶│ Drafter(s)│────▶│ Verifier │────▶│  Fuser  │
│ (1 call) │     │ (k calls) │     │ (k calls)│     │ (merge) │
└─────────┘     └──────────┘     └──────────┘     └─────────┘
    │                │                │                │
    ▼                ▼                ▼                ▼
 Numbered         Candidate       Score +          Final
 Plan + DoD       Answers         Issues           Answer
```

### Components
- **Planner**: Converts user goal into numbered plan + Definition of Done
- **Drafter(s)**: Generates k candidate answers following the plan
- **Verifier**: Scores candidates on factuality, completeness, clarity; flags issues
- **Fuser**: Merges best candidates into polished final answer
- **Logger**: Structured JSON logs of all prompts, responses, and scores

## Models & Licensing

| Model | License | Size | Notes |
|-------|---------|------|-------|
| Mistral-7B-Instruct-v0.3 | Apache 2.0 | 7B | Recommended primary |
| Llama-2-7B-Chat | Llama 2 Community | 7B | Alternative |
| Qwen2-7B-Instruct | Apache 2.0 | 7B | Alternative |
| TinyLlama-1.1B-Chat | Apache 2.0 | 1.1B | CPU fallback |

**All models used are licensed for public use.** See individual model cards for full license terms.

## Quantization

Produce CPU-runnable quantized models:

```bash
# Using llama.cpp (recommended)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
# Convert HF model to GGUF
python convert_hf_to_gguf.py mistralai/Mistral-7B-Instruct-v0.3 --outfile mistral-7b-instruct.gguf
# Quantize to Q4_0
./quantize mistral-7b-instruct.gguf mistral-7b-instruct-q4_0.bin q4_0
```

See `docs/QUANTIZATION.md` for GPTQ and AWQ paths.

## Evaluation

```bash
# Run automated evaluation
python -m tests.eval_harness --prompts tests/sample_prompts.jsonl

# Metrics produced:
# - Verifier pass rate (target: >= 60%)
# - Average plan length
# - Candidate diversity score
# - Hallucination incidence
# - Average latency per query
```

## LoRA Fine-Tuning

```bash
# On Colab with GPU
python src/train_lora.py \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/instructions.jsonl \
  --epochs 2 \
  --lora-r 8 \
  --lora-alpha 16 \
  --batch-size 8
```

## Project Structure

```
fable5-pipeline/
├── src/
│   ├── __init__.py
│   ├── orchestrator.py          # Core orchestrator (plan→draft→verify→fuse)
│   ├── inference_backends.py    # llama_cpp_runner, hf_bnb_runner
│   └── prompts.py               # System prompt templates
├── tests/
│   ├── eval_harness.py          # Automated evaluation
│   ├── sample_prompts.jsonl     # 20 evaluation prompts
│   └── test_orchestrator.py     # Unit tests
├── notebooks/
│   └── colab_starter.ipynb      # One-click Colab demo
├── examples/
│   ├── run_local.sh             # Local CPU demo script
│   └── run_colab.md             # Colab instructions
├── data/
│   └── instructions.jsonl       # Sample training data for LoRA
├── docs/
│   └── QUANTIZATION.md          # Detailed quantization guide
├── requirements.txt
├── ETHICS.md
└── README.md
```

## Safety & Legal

- **No proprietary weights or leaked prompts** are used or distributed.
- **No paywalled APIs** required — everything runs on free local hardware or free Colab tier.
- All model weights used have explicit licenses permitting downstream use.
- See [ETHICS.md](ETHICS.md) for responsible use guidelines.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run tests: `python -m pytest tests/`
4. Submit a PR

## License

MIT License — see [LICENSE](LICENSE) for details.
