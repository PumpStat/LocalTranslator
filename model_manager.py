#!/usr/bin/env python3
"""
Model/GPU management and inference helpers for the translation server.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore

if TYPE_CHECKING:
    from text_processing import TextProcessor

NLLB_CODES = {"ko": "kor_Hang", "en": "eng_Latn", "es": "spa_Latn"}


def _get_env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class ModelManager:
    GPU_MODEL_PRESETS = {
        # LLM presets (only used when BACKEND=llm)
        "3080": {"llm": "Qwen/Qwen2.5-7B-Instruct"},
        "5090": {"llm": "Qwen/Qwen2.5-32B-Instruct"},
        # NLLB presets (used when BACKEND=nllb)
        "3080_nllb": {"mt": "facebook/nllb-200-distilled-1.3B"},
        "5090_nllb": {"mt": "facebook/nllb-200-3.3B"},
    }

    def __init__(self) -> None:
        self.gpu_profile = os.getenv("GPU_PROFILE", "").strip().lower()
        self.backend = os.getenv("BACKEND", "llm").strip().lower()
        self.model_name = os.getenv("MODEL_NAME")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "1024") or 1024)
        self.gpu_info = self._detect_gpu_info()
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = self._has_mps()
        self.device_hint = "cuda" if self.has_cuda else ("mps" if self.has_mps else "cpu")
        self.device_map = "auto" if (self.has_cuda or self.has_mps) else None
        self.use_4bit: Optional[bool] = None
        self.tokenizer = None
        self.model = None
        self._load_backend()

    # ----- GPU helpers -----
    @staticmethod
    def _detect_gpu_info() -> Dict[str, Optional[str]]:
        name = None
        vram_gb = None
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                name = props.name
                vram_gb = round(props.total_memory / (1024**3))
            except Exception:
                pass
        return {"name": name, "vram_gb": vram_gb}

    @staticmethod
    def _has_mps() -> bool:
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend:
            return False
        try:
            return bool(mps_backend.is_available() and mps_backend.is_built())
        except AttributeError:
            return bool(mps_backend.is_available())

    def _preferred_torch_dtype(self) -> torch.dtype:
        if self.has_cuda:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if self.has_mps:
            return torch.float16
        return torch.float32

    # ----- Loading logic -----
    def _load_backend(self) -> None:
        if self.backend == "nllb":
            self._load_nllb()
        else:
            self.backend = "llm"
            self._load_llm()

    def _select_llm_name(self) -> str:
        if self.model_name:
            return self.model_name
        preset = self.GPU_MODEL_PRESETS.get(self.gpu_profile) if self.gpu_profile else None
        if preset and "llm" in preset:
            return preset["llm"]
        vram = self.gpu_info.get("vram_gb") or 0
        if vram >= 30:
            return "Qwen/Qwen2.5-32B-Instruct"
        if vram >= 18:
            return "Qwen/Qwen2.5-14B-Instruct"
        return "Qwen/Qwen2.5-7B-Instruct"

    def _select_mt_name(self) -> str:
        if self.model_name:
            return self.model_name
        preset = (
            self.GPU_MODEL_PRESETS.get(f"{self.gpu_profile}_nllb") if self.gpu_profile else None
        )
        if preset and "mt" in preset:
            return preset["mt"]
        vram = self.gpu_info.get("vram_gb") or 0
        return "facebook/nllb-200-3.3B" if vram >= 20 else "facebook/nllb-200-distilled-1.3B"

    def _load_llm(self) -> None:
        model_name = self._select_llm_name()
        self.model_name = model_name
        self.use_4bit = _get_env_bool("USE_4BIT", self.has_cuda)
        quant_cfg = None
        if self.use_4bit:
            if not self.has_cuda:
                print("[Model] Requested 4-bit quantization but CUDA is unavailable; falling back to full precision.")
                self.use_4bit = False
            elif BitsAndBytesConfig is None:
                print("[Model] bitsandbytes is missing; disabling 4-bit quantization.")
                self.use_4bit = False
            else:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if (self.has_cuda and torch.cuda.is_bf16_supported())
                        else torch.float16
                    ),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        dtype = None if self.use_4bit else self._preferred_torch_dtype()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        device_desc = self.device_hint
        if self.use_4bit:
            device_desc = f"{device_desc}+4bit"
        print(
            f"[Model] BACKEND=llm MODEL_NAME={model_name}, USE_4BIT={self.use_4bit}, "
            f"DEVICE={device_desc}, GPU_PROFILE={self.gpu_profile}, CUDA_GPU={self.gpu_info}"
        )

    def _load_nllb(self) -> None:
        model_name = self._select_mt_name()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        dtype = torch.float16 if (self.has_cuda or self.has_mps) else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        print(
            f"[Model] BACKEND=nllb MODEL_NAME={model_name} GPU_PROFILE={self.gpu_profile} "
            f"CUDA_GPU={self.gpu_info} DEVICE={self.device_hint}"
        )

    # ----- Inference helpers -----
    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.backend != "llm":
            raise RuntimeError("LLM backend is not initialized.")
        assert self.tokenizer is not None and self.model is not None
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0, input_len:]
        txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return txt

    def translate_nllb(
        self,
        text: str,
        src: str,
        tgt: str,
        processor: "TextProcessor",
    ) -> str:
        if self.backend != "nllb":
            raise RuntimeError("NLLB backend is not initialized.")
        assert self.tokenizer is not None and self.model is not None
        t_in, placeholders, placeholder_tokens = processor.apply_placeholders(text, src, tgt)
        sanitized = processor.sanitize_prompt(t_in)
        src_code = NLLB_CODES[src]
        tgt_code = NLLB_CODES[tgt]
        try:
            self.tokenizer.src_lang = src_code
        except Exception:
            pass
        inputs = self.tokenizer(sanitized, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        forced_bos = self._forced_bos_token_id(self.tokenizer, tgt_code)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos
        gen = self.model.generate(**inputs, **gen_kwargs)
        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        out = processor.ensure_placeholder_tokens(out, placeholder_tokens)
        out = processor.restore_placeholders(out, placeholders)
        if tgt == "es":
            out = out.replace("-", " ").replace("_", " ")
        out = processor.postprocess_output(out, tgt, text)
        return out

    @staticmethod
    def _forced_bos_token_id(tokenizer, code: str) -> Optional[int]:
        bos_id = None
        if hasattr(tokenizer, "lang_code_to_id"):
            try:
                bos_id = tokenizer.lang_code_to_id[code]
            except Exception:
                bos_id = None
        if bos_id is None and hasattr(tokenizer, "get_lang_id"):
            try:
                bos_id = tokenizer.get_lang_id(code)
            except Exception:
                bos_id = None
        if bos_id is None:
            for candidate in (code, f"__{code}__", f"<{code}>"):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(candidate)
                    if isinstance(token_id, int) and token_id > 0:
                        bos_id = token_id
                        break
                except Exception:
                    continue
        return bos_id
