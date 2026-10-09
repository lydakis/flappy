"""Small pretrained chat model with a LoRA adapter, trained in place on MPS/CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from student.grpo import grpo_loss

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a careful assistant. Follow the requested output format exactly "
    "and do not add explanations unless asked."
)


class StudentModel(Protocol):
    def generate(
        self,
        prompts: list[str],
        *,
        sample: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def train_step(self, examples: list[tuple[str, str]]) -> float: ...

    def sample_group(
        self, prompt: str, n: int, max_new_tokens: int | None = None
    ) -> list[str]: ...

    def policy_step(
        self,
        prompt: str,
        completions: list[str],
        advantages: list[float],
        kl_coef: float,
    ) -> dict: ...

    def hybrid_step(
        self,
        group: tuple[str, list[str], list[float]] | None,
        replay: list[tuple[str, str, float]],
        kl_coef: float,
    ) -> dict: ...


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

    def generate(
        self,
        prompts: list[str],
        *,
        sample: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        torch, tok = self.torch, self.tokenizer
        self.model.eval()
        tok.padding_side = "left"
        batch = tok(
            [self._chat(p) for p in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        kwargs = {
            "do_sample": sample,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
        }
        if sample:
            kwargs.update(temperature=self.temperature, top_p=0.95)
        with torch.no_grad():
            out = self.model.generate(**batch, pad_token_id=tok.pad_token_id, **kwargs)
        width = batch["input_ids"].shape[1]
        return [tok.decode(row[width:], skip_special_tokens=True) for row in out]

    def _batch(self, examples: list[tuple[str, str]]):
        """Right-padded ids, attention mask and completion-only labels."""
        torch, tok = self.torch, self.tokenizer
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
        return input_ids.to(self.device), mask.to(self.device), labels.to(self.device)

    def _step(self, loss) -> None:
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train_step(self, examples: list[tuple[str, str]]) -> float:
        """One SFT step on prompt/completion pairs; loss on completions only."""
        self.model.train()
        logp, keep = self._completion_logps(*self._batch(examples))
        loss = -logp.sum() / keep.sum()
        self._step(loss)
        return float(loss.detach().cpu())

    def sample_group(
        self, prompt: str, n: int, max_new_tokens: int | None = None
    ) -> list[str]:
        return self.generate([prompt] * n, sample=True, max_new_tokens=max_new_tokens)

    def _completion_logps(self, input_ids, mask, labels):
        """Log-probs of completion tokens as [batch, tokens], zero elsewhere.

        Only completion positions go through the LM head: full-vocabulary logits
        over whole prompts pushed a 16 GB machine into swap.
        """
        base = self.model.get_base_model()
        hidden = base.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
        targets = labels[:, 1:]
        keep = targets != -100
        logits = base.lm_head(hidden[:, :-1][keep]).float()
        picked = logits.gather(-1, targets[keep].unsqueeze(-1)).squeeze(-1)
        selected = picked - logits.logsumexp(dim=-1)
        logp = selected.new_zeros(targets.shape).masked_scatter(keep, selected)
        return logp, keep.float()

    def _grpo_term(self, prompt: str, completions: list[str], advantages, kl_coef):
        torch = self.torch
        input_ids, mask, labels = self._batch([(prompt, c) for c in completions])
        logp, keep = self._completion_logps(input_ids, mask, labels)
        with torch.no_grad(), self.model.disable_adapter():
            ref_logp, _ = self._completion_logps(input_ids, mask, labels)
        adv = torch.tensor(advantages, dtype=logp.dtype, device=logp.device)
        return grpo_loss(logp, ref_logp, adv, keep, kl_coef)

    def policy_step(
        self,
        prompt: str,
        completions: list[str],
        advantages: list[float],
        kl_coef: float,
    ) -> dict:
        """One GRPO step on a sampled group; the reference is the adapter-free base."""
        self.model.train()
        loss, kl = self._grpo_term(prompt, completions, advantages, kl_coef)
        self._step(loss)
        return {"loss": float(loss.detach().cpu()), "kl": float(kl.detach().cpu())}

    def hybrid_step(
        self,
        group: tuple[str, list[str], list[float]] | None,
        replay: list[tuple[str, str, float]],
        kl_coef: float,
    ) -> dict:
        """One step on GRPO(group) + weighted SFT(replay); either part may be absent.

        Replay loss is each example's token-mean NLL times its weight, averaged.
        """
        torch = self.torch
        self.model.train()
        grpo = kl = sft = torch.zeros((), device=self.device)
        if group is not None:
            grpo, kl = self._grpo_term(*group, kl_coef)
        if replay:
            batch = self._batch([(p, c) for p, c, _ in replay])
            logp, keep = self._completion_logps(*batch)
            nll = -logp.sum(dim=1) / keep.sum(dim=1).clamp(min=1)
            weights = torch.tensor([w for *_, w in replay], device=nll.device)
            sft = (weights * nll).mean()
        loss = grpo + sft
        if loss.requires_grad:
            self._step(loss)
        return {
            "loss": float(loss.detach().cpu()),
            "grpo": float(grpo.detach().cpu()),
            "sft": float(sft.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }

    def save_adapter(self, path: Path) -> None:
        self.model.save_pretrained(str(path))
