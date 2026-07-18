"""Chat-format text generation with an instruction-tuned model."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Generator:
    """Wraps a transformers chat model behind a simple ``chat(messages)`` call.

    Uses the model's chat template (instead of raw string prompts), greedy
    decoding for reproducibility, fp16 on GPU, and ``torch.inference_mode``.
    """

    def __init__(self, model_name: str, max_new_tokens: int):
        import torch
        from transformers import pipeline

        from nutriguide.models import best_device

        device = best_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading generation model %s on %s (%s)", model_name, device, dtype)
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=dtype,
        )
        self._max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int | None = None) -> str:
        import torch

        with torch.inference_mode():
            out = self._pipe(
                messages,
                max_new_tokens=max_new_tokens or self._max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        return out[0]["generated_text"].strip()
