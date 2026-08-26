"""Base model + LoRA + tokenizer loading, shared by every training/eval entrypoint.

Stage 0 (see BUILD_PLAN.md section 3). This is the one module that's fully
implemented at scaffold time -- `python -m model.load` is the Stage 0
acceptance check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj")
    )


def load_tokenizer(base_model: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_model: str, load_in_4bit: bool = False, device_map: str = "auto"):
    """Load the frozen base causal LM, optionally in 4-bit (bitsandbytes)."""
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def attach_lora(
    model,
    settings: Optional[LoraSettings] = None,
    adapter_path: Optional[str] = None,
):
    """Attach a fresh LoRA adapter, or load a trained one from disk."""
    if adapter_path is not None:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    settings = settings or LoraSettings()
    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_policy_and_tokenizer(
    base_model: str,
    load_in_4bit: bool = False,
    adapter_path: Optional[str] = None,
    lora_settings: Optional[LoraSettings] = None,
):
    """Convenience entrypoint: base + LoRA (fresh or loaded) + tokenizer.

    The DPO reference model is NOT a second copy in memory -- disable the
    adapter on this same PeftModel (`model.disable_adapter()`) to recover
    reference (base) behaviour. See src/train/dpo.py and CLAUDE.md.
    """
    tokenizer = load_tokenizer(base_model)
    base = load_base_model(base_model, load_in_4bit=load_in_4bit)
    model = attach_lora(base, settings=lora_settings, adapter_path=adapter_path)
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test model loading (Stage 0 acceptance check).")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_policy_and_tokenizer(args.base_model, load_in_4bit=args.load_in_4bit)
    print(f"Loaded {args.base_model} with a fresh LoRA adapter.")
    model.print_trainable_parameters()
