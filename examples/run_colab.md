# Running on Google Colab

## Quick Start (Free T4 GPU)

1. Open `notebooks/colab_starter.ipynb` in Colab
2. Run all cells — installs dependencies, loads model, runs demo

## Manual Steps

### Step 1: Install Dependencies
```python
!pip install transformers accelerate bitsandbytes peft sentence-transformers faiss-cpu tqdm pydantic
```

### Step 2: Clone Repo
```python
!git clone https://github.com/yourname/fable5-pipeline.git
%cd fable5-pipeline
```

### Step 3: Run Demo
```python
from src.orchestrator import Orchestrator
from src.inference_backends import HfBnbRunner

backend = HfBnbRunner(model_id="mistralai/Mistral-7B-Instruct-v0.3")
orch = Orchestrator(backend, k_candidates=2)

trace = orch.run("Explain the concept of supply and demand with real-world examples.")
print("FINAL ANSWER:")
print(trace["final_answer"])
print("\nMETRICS:")
print(trace["metrics"])
```

## Free Tier Limits
- **T4 GPU**: ~15GB VRAM — fits 7B 4-bit model
- **Runtime**: ~2-5 min per prompt (3 candidates)
- **Session**: 12 hours max, reconnects if disconnected
- **Storage**: Temporary — download results before session ends

## Tips
- Use `!nvidia-smi` to verify GPU availability
- Use TinyLlama (1.1B) if T4 is unavailable
- Save logs to Google Drive for persistence
