# Agentic AI + XAI on CSIC 2010

A small LangChain agent that acts as a web-application-firewall analyst. It reads raw HTTP
requests from the CSIC 2010 dataset and calls tools served by two **MCP servers**. The tools are
stubs: they only report that they were triggered. Which tool the agent chooses, with which
arguments, and why, is the behaviour we will explain later with XAI techniques.

The LLM is **meta-llama/Llama-3.1-8B-Instruct**, run locally with Hugging Face `transformers`
so that its internals (attention, gradients, logits) are available for XAI. It is downloaded
when the program starts (first run only) and needs a CUDA GPU, so it is meant to run on Google Colab.

```
main.py ──► agent_xai/agent.py  (LangChain create_agent)
               │  agent_xai/llama_model.py   download + load Llama 3.1, LangChain chat-model wrapper
               │                             (native Llama 3.1 tool calling, greedy decoding)
               │  langchain-mcp-adapters (stdio, persistent sessions)
               ├──► mcp_servers/analysis_server.py   detect_sql_injection, detect_xss,
               │                                     detect_path_traversal, check_parameter_anomaly
               └──► mcp_servers/response_server.py   allow_request, block_request, raise_alert
            agent_xai/data_loader.py  parses CSIC raw files -> HttpRequest (with ground-truth label)
            agent_xai/tracing.py      writes one JSON trace per request to runs/*.jsonl
```

## Run on Google Colab

Open [colab_run.ipynb](colab_run.ipynb) in Colab and follow its cells. In short:

1. Use a GPU runtime (T4 is enough: the model loads in 4-bit, about 6 GB of GPU memory).
2. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and add your
   Hugging Face token as the Colab secret `HF_TOKEN`.
3. Put this folder on Google Drive, then:

```bash
pip install -r requirements-colab.txt
python main.py --check-tools                        # no LLM: start both MCP servers, call each tool
python main.py --n 4                                # downloads + loads Llama, then 4 random requests
python main.py --dataset anomalous_test --index 0   # one specific request
```

Settings (environment variables or `.env`): `LLAMA_QUANTIZATION` (`auto` = full 16-bit on GPUs
with 20 GB or more, else 4-bit), `MAX_NEW_TOKENS`, `RECURSION_LIMIT`, `LLM_MODEL`.

## XAI: LIME

`explain.py` explains *why* the agent made a decision, with LIME
([agent_xai/xai/](agent_xai/xai/)):

1. **Features** ([features.py](agent_xai/xai/features.py)): the request is split into readable
   features, either every token in the URL path and parameter values (`--granularity token`)
   or every parameter (`--granularity param`). Headers, method and host stay fixed.
2. **Perturb**: random features are deleted, and the request is re-encoded as valid HTTP.
3. **Black box** ([probe.py](agent_xai/xai/probe.py)): Llama gets the agent's exact system
   prompt, MCP tool schemas and the perturbed request. The probe forces the start of the answer
   and reads the probability of each outcome in one forward pass:
   `verdict` (normal/attack), `action` (allow/block) or `first_tool` (which tool it picks first).
   The tools are stubs, so the request is the agent's only evidence. With `--trace`, the script
   reports how often the probe agrees with what the agent actually did.
4. **Surrogate** ([lime.py](agent_xai/xai/lime.py)): weighted Ridge regression (the standard
   LIME text kernel). Each weight is how much keeping that feature changes the probability of
   the decision. Check the R² before trusting the weights.

```bash
python explain.py --dataset anomalous_test --index 1                       # one request
python explain.py --trace runs/run_<...>.jsonl --target action --granularity param
```

Each explanation is saved as `runs/lime/<request>_<target>_<granularity>.html` (the request
with each token coloured by its weight, plus a bar chart) and as `.json` for later analysis.

## Local machine

`uv sync` installs everything except the model stack; `python main.py --check-tools` works
without a GPU. `uv sync --extra llama` adds transformers/bitsandbytes (a CUDA GPU is still needed).

## Trace format (input for the XAI phase)

Each line of `runs/run_*.jsonl` contains: `request_id`, `ground_truth`, `predicted`, `correct`,
`verdict` (the agent's JSON with `attack_type`, `confidence`, `evidence`), `tool_sequence`, the
full parsed `request`, and every LLM/tool step in `steps` (including Llama's raw output).
