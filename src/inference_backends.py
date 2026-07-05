"""Inference backends for the Fable5 pipeline.

Provides two backends:
1. llama_cpp_runner — runs quantized GGUF models via llama.cpp CLI
2. hf_bnb_runner — runs HuggingFace models with 4-bit bitsandbytes quantization (GPU)
"""

import json
import subprocess
import time
import os
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class InferenceResult:
    text: str
    latency_ms: float
    model: str
    backend: str
    tokens_generated: int = 0
    error: Optional[str] = None


class LlamaCppRunner:
    """Run inference using llama.cpp CLI with quantized GGUF models."""

    def __init__(
        self,
        model_path: str,
        llama_cpp_bin: str = "./llama.cpp/main",
        n_ctx: int = 2048,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        threads: int = 4,
    ):
        self.model_path = model_path
        self.llama_cpp_bin = llama_cpp_bin
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.threads = threads

    def generate(self, system_prompt: str, user_prompt: str) -> InferenceResult:
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        cmd = [
            self.llama_cpp_bin,
            "-m", self.model_path,
            "-p", full_prompt,
            "-n", str(self.max_tokens),
            "--temp", str(self.temperature),
            "-c", str(self.n_ctx),
            "-t", str(self.threads),
            "--no-display-prompt",
        ]
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            latency = (time.time() - start) * 1000
            output = proc.stdout.strip()
            tokens = len(output.split())
            return InferenceResult(
                text=output,
                latency_ms=latency,
                model=os.path.basename(self.model_path),
                backend="llama_cpp",
                tokens_generated=tokens,
            )
        except subprocess.TimeoutExpired:
            return InferenceResult(
                text="",
                latency_ms=(time.time() - start) * 1000,
                model=os.path.basename(self.model_path),
                backend="llama_cpp",
                error="Timeout after 120s",
            )
        except FileNotFoundError:
            return InferenceResult(
                text="",
                latency_ms=0,
                model=os.path.basename(self.model_path),
                backend="llama_cpp",
                error=f"llama.cpp binary not found: {self.llama_cpp_bin}",
            )


class HfBnbRunner:
    """Run inference using HuggingFace transformers with 4-bit bitsandbytes quantization.

    Requires: pip install transformers accelerate bitsandbytes
    Requires: CUDA GPU (works on free Colab T4)
    """

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device_map,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> InferenceResult:
        self._load_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        start = time.time()
        try:
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
            latency = (time.time() - start) * 1000
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            result_text = self._tokenizer.decode(generated, skip_special_tokens=True)
            tokens = len(generated)
            return InferenceResult(
                text=result_text,
                latency_ms=latency,
                model=self.model_id,
                backend="hf_bnb",
                tokens_generated=tokens,
            )
        except Exception as e:
            return InferenceResult(
                text="",
                latency_ms=(time.time() - start) * 1000,
                model=self.model_id,
                backend="hf_bnb",
                error=str(e),
            )


def create_backend(backend_type: str, **kwargs):
    """Factory function to create an inference backend."""
    if backend_type == "llama_cpp":
        return LlamaCppRunner(**kwargs)
    elif backend_type == "hf_bnb":
        return HfBnbRunner(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Use 'llama_cpp' or 'hf_bnb'.")
