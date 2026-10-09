"""Small pretrained chat model with a LoRA adapter, trained in place on MPS/CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a careful assistant. Follow the requested output format exactly "
    "and do not add explanations unless asked."
)


class StudentModel(Protocol):
    def generate(self, prompts: list[str], *, sample: bool = False) -> list[str]: ...

    def train_step(self, examples: list[tuple[str, str]]) -> float: ...


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class HFStudent:
    """Hugging Face causal LM + peft LoRA. Only adapter weights are trained."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        lora_rank: int = 16,
        lr: float = 1e-4,
        max_new_tokens: int = 256,
        max_train_tokens: int = 768,
        temperature: float = 0.7,
        seed: int = 0,
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )
        self.max_new_tokens = max_new_tokens
        self.max_train_tokens = max_train_tokens
        self.temperature = temperature

    def _chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompts: list[str], *, sample: bool = False) -> list[str]:
        torch, tok = self.torch, self.tokenizer
        self.model.eval()
        tok.padding_side = "left"
        batch = tok(
            [self._chat(p) for p in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        kwargs = {"do_sample": sample, "max_new_tokens": self.max_new_tokens}
        if sample:
            kwargs.update(temperature=self.temperature, top_p=0.95)
        with torch.no_grad():
            out = self.model.generate(**batch, pad_token_id=tok.pad_token_id, **kwargs)
        width = batch["input_ids"].shape[1]
        return [tok.decode(row[width:], skip_special_tokens=True) for row in out]

    def train_step(self, examples: list[tuple[str, str]]) -> float:
        """One optimizer step on prompt/completion pairs; loss on completions only."""
        torch, tok = self.torch, self.tokenizer
        self.model.train()
        rows = []
        for prompt, completion in examples:
            head = tok(self._chat(prompt), add_special_tokens=False)["input_ids"]
            tail = tok(completion.strip() + tok.eos_token, add_special_tokens=False)[
                "input_ids"
            ]
            ids = (head + tail)[: self.max_train_tokens]
            labels = ([-100] * len(head) + tail)[: self.max_train_tokens]
            rows.append((ids, labels))
        width = max(len(ids) for ids, _ in rows)
        pad = tok.pad_token_id
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in rows])
        labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in rows])
        mask = torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows]
        )
        loss = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=mask.to(self.device),
            labels=labels.to(self.device),
        ).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach().cpu())

    def save_adapter(self, path: Path) -> None:
        self.model.save_pretrained(str(path))
