"""Local Llama-3.1-8B-Instruct as a LangChain chat model (Hugging Face transformers).

* The weights are downloaded from the Hugging Face Hub when the program starts
  (`load_llama`), then reused from the HF cache on later runs.
* Tool calling uses Llama 3.1's native format: the chat template injects the tool
  JSON schemas, and the model answers with {"name": ..., "parameters": {...}}.
* Generation is greedy (deterministic), which keeps runs reproducible for XAI.
* The raw model and tokenizer stay accessible (`LlamaChatModel.hf_model` /
  `.tokenizer`) so later XAI code can get attention, gradients and logits.

Needs a CUDA GPU (e.g. Google Colab T4). The repo is gated: accept Meta's license on
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and provide HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

# ---------------------------------------------------------------------------
# Download + load
# ---------------------------------------------------------------------------


def _hf_token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:  # Colab "Secrets" panel
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def download_model(model_id: str) -> str:
    """Download the safetensors weights + tokenizer (skips Meta's duplicate original/*.pth)."""
    from huggingface_hub import snapshot_download

    token = _hf_token()
    if token is None:
        raise RuntimeError(
            "HF_TOKEN not found. Accept the license at "
            f"https://huggingface.co/{model_id} and set HF_TOKEN (env var or Colab secret)."
        )
    print(f"[llama] Downloading / verifying {model_id} (first run ~16 GB) ...", flush=True)
    return snapshot_download(
        model_id,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
        ignore_patterns=["original/*"],
    )


def load_llama(model_id: str, quantization: str = "auto", max_new_tokens: int = 512) -> "LlamaChatModel":
    """Download (if needed) and load the model onto the GPU.

    quantization: "auto" | "4bit" | "8bit" | "none".
        auto -> full 16-bit if the GPU has >= 20 GB (A100 / L4), else 4-bit NF4 (T4, ~6 GB).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")

    path = download_model(model_id)

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if quantization == "auto":
        quantization = "none" if gpu_gb >= 20 else "4bit"

    quant_config = None
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"[llama] Loading on {torch.cuda.get_device_name(0)} ({gpu_gb:.0f} GB), "
          f"quantization={quantization}, dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=dtype,
        quantization_config=quant_config,
    )
    model.eval()
    print("[llama] Model ready.", flush=True)
    return LlamaChatModel(hf_model=model, tokenizer=tokenizer, model_id=model_id,
                          max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# LangChain <-> Llama 3.1 chat format
# ---------------------------------------------------------------------------


def to_llama_chat(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to the dicts Llama 3.1's chat template expects."""
    chat: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            chat.append({"role": "system", "content": m.text})
        elif isinstance(m, HumanMessage):
            chat.append({"role": "user", "content": m.text})
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                # The Llama 3.1 template accepts exactly one tool call per assistant turn.
                for tc in m.tool_calls:
                    chat.append({
                        "role": "assistant",
                        "tool_calls": [{"type": "function",
                                        "function": {"name": tc["name"], "arguments": tc["args"]}}],
                    })
            else:
                # Note: the key 'tool_calls' must be absent (not empty) for plain replies.
                chat.append({"role": "assistant", "content": m.text})
        elif isinstance(m, ToolMessage):
            chat.append({"role": "tool", "content": m.text})
        else:
            raise TypeError(f"Unsupported message type: {type(m).__name__}")
    return chat


def parse_tool_call(text: str, tool_names: set[str]) -> tuple[dict, str] | None:
    """Find the first {"name": <known tool>, "parameters": {...}} object in model output.

    Returns (tool_call, text_before_it) or None if the output is a normal reply.
    """
    text = text.replace("<|python_tag|>", "")
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in tool_names:
            args = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            call = {"name": obj["name"], "args": args or {}, "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call"}
            return call, text[:i].strip()
        i = text.find("{", end)
    return None


class LlamaChatModel(BaseChatModel):
    """LangChain chat model backed by a local transformers Llama 3.1 model."""

    hf_model: Any
    tokenizer: Any
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "local-llama-3.1"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # tool_choice is ignored: Llama decides itself whether to call a tool.
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    # Separated out so it can be swapped/monkeypatched (tests) or hooked (XAI).
    def generate_text(self, chat: list[dict], tools: list[dict] | None) -> tuple[str, dict]:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            chat, tools=tools or None, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.hf_model.device)
        with torch.no_grad():
            out = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,  # greedy = reproducible
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        usage = {"input_tokens": int(prompt_len), "output_tokens": int(len(new_tokens)),
                 "total_tokens": int(prompt_len + len(new_tokens))}
        return text, usage

    def _generate(self, messages, stop=None, run_manager=None, tools=None, **kwargs) -> ChatResult:
        chat = to_llama_chat(messages)
        text, usage = self.generate_text(chat, tools)

        tool_names = {t["function"]["name"] for t in tools or []}
        parsed = parse_tool_call(text, tool_names) if tool_names else None
        meta = {"model": self.model_id, "raw_output": text}
        if parsed:
            call, preamble = parsed
            msg = AIMessage(content=preamble, tool_calls=[call], response_metadata=meta,
                            usage_metadata=usage)
        else:
            msg = AIMessage(content=text, response_metadata=meta, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=msg)])
