"""Decision probe: P(agent decision | request) in one forward pass of Llama.

Why a probe instead of re-running the agent?  LIME needs hundreds of model calls per
explanation. A full agent run is several generate() calls (minutes on a T4), whereas a
probe is one forward pass (~1 s). The probe shows Llama *exactly* the agent's context -
same system prompt, same MCP tool schemas, same request formatting - then forces the start
of the answer and reads the probabilities of the possible continuations:

  target="verdict"     assistant starts  {"verdict": "      -> normal | attack
  target="action"      assistant starts  {"name": "         -> allow_request | block_request
  target="first_tool"  assistant starts  {"name": "         -> any of the 7 tool names

Because the MCP tools are stubs (they add no information), the request text is the only
evidence the agent has, so this probe is a faithful proxy for the decision. Check it
anyway: `explain.py --trace ...` reports how often the probe agrees with the agent.

Each class score is the full log-likelihood of its token sequence; scores are then
normalised over the candidate set. `mass` = how much raw probability the candidates had
(low mass means the model wanted to say something else at that point).
"""

from __future__ import annotations

import numpy as np

from ..agent import SYSTEM_PROMPT, request_prompt
from ..data_loader import HttpRequest

TARGETS = {
    "verdict": ('{"verdict": "', ["normal", "attack"]),
    "action": ('{"name": "', ["allow_request", "block_request"]),
    "first_tool": ('{"name": "', None),  # None = all tool names
}


class DecisionProbe:
    def __init__(self, llm, tool_schemas: list[dict], target: str = "verdict", batch_size: int = 4):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {list(TARGETS)}")
        if not hasattr(llm, "hf_model"):
            raise TypeError("The probe needs the local Hugging Face model (LlamaChatModel).")
        prefix, classes = TARGETS[target]
        self.target = target
        self.tools = tool_schemas
        self.class_names = classes or [t["function"]["name"] for t in tool_schemas]
        self.model, self.tok = llm.hf_model, llm.tokenizer
        self.batch_size = batch_size

        self.prefix_ids = self.tok.encode(prefix, add_special_tokens=False)
        self.cand_ids = [self.tok.encode(c, add_special_tokens=False) for c in self.class_names]
        self.single_token = all(len(c) == 1 for c in self.cand_ids)
        self._cache: dict[str, tuple[np.ndarray, float]] = {}

    def prompt_ids(self, req: HttpRequest) -> list[int]:
        chat = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request_prompt(req)}]
        text = self.tok.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True,
                                            tokenize=False)
        return self.tok.encode(text, add_special_tokens=False) + self.prefix_ids

    # ------------------------------------------------------------------------------

    def _forward_logprobs(self, batch: list[list[int]], keep: int):
        """Left-padded batch forward; returns log-probs of the last `keep` positions."""
        import torch

        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        L = max(len(s) for s in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for j, s in enumerate(batch):
            ids[j, L - len(s):] = torch.tensor(s)
            attn[j, L - len(s):] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        dev = self.model.device
        out = self.model(input_ids=ids.to(dev), attention_mask=attn.to(dev),
                         position_ids=pos.to(dev), logits_to_keep=keep)
        return torch.log_softmax(out.logits.float(), dim=-1).cpu()  # (B, keep, V)

    def _score(self, prompts: list[list[int]]) -> np.ndarray:
        """(n, K) log-probabilities of each candidate sequence."""
        import torch

        n, K = len(prompts), len(self.class_names)
        scores = np.zeros((n, K))
        bs = self.batch_size
        with torch.no_grad():
            if self.single_token:  # one forward per request, read next-token distribution
                first = [c[0] for c in self.cand_ids]
                for b in range(0, n, bs):
                    lp = self._forward_logprobs(prompts[b:b + bs], keep=1)
                    scores[b:b + bs] = lp[:, -1, first].numpy()
            else:  # one forward per (request, candidate), teacher-forcing the candidate
                jobs = [(i, k, prompts[i] + c) for i in range(n) for k, c in enumerate(self.cand_ids)]
                keep = max(len(c) for c in self.cand_ids) + 1
                for b in range(0, len(jobs), bs):
                    chunk = jobs[b:b + bs]
                    lp = self._forward_logprobs([s for _, _, s in chunk], keep=keep)
                    for j, (i, k, _) in enumerate(chunk):
                        c = self.cand_ids[k]
                        # token at position -m is predicted by the logits at position -m-1
                        rows = torch.arange(keep - len(c) - 1, keep - 1)
                        scores[i, k] = lp[j, rows, torch.tensor(c)].sum().item()
        return scores

    def predict(self, reqs: list[HttpRequest]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs (n,K) normalised over candidates, mass (n,) raw candidate prob)."""
        keys = [r.to_raw() for r in reqs]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        uniq = list(dict.fromkeys(keys[i] for i in todo))
        if uniq:
            first_idx = {k: keys.index(k) for k in uniq}
            logp = self._score([self.prompt_ids(reqs[first_idx[k]]) for k in uniq])
            for k, row in zip(uniq, logp):
                mass = float(np.exp(row).sum())
                p = np.exp(row - row.max())
                self._cache[k] = (p / p.sum(), mass)
        probs = np.stack([self._cache[k][0] for k in keys])
        mass = np.array([self._cache[k][1] for k in keys])
        return probs, mass
