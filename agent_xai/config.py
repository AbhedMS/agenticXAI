"""Central configuration: paths and model selection."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = (
    PROJECT_ROOT
    / "Web-Application-Attack-Datasets-master"
    / "OriginalDataSets"
    / "csic_2010"
)

# Logical dataset name -> (file, ground-truth label)
DATASETS = {
    "normal_train": (DATA_DIR / "normalTrafficTraining.txt", "normal"),
    "normal_test": (DATA_DIR / "normalTrafficTest.txt", "normal"),
    "anomalous_test": (DATA_DIR / "anomalousTrafficTest.txt", "anomalous"),
}

RUNS_DIR = PROJECT_ROOT / "runs"

# A Hugging Face repo id ("org/name") runs locally with transformers (default).
# A provider string such as "anthropic:claude-opus-5-5" goes through LangChain's init_chat_model.
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Local model settings: quantization = auto | 4bit | 8bit | none
LLAMA_QUANTIZATION = os.getenv("LLAMA_QUANTIZATION", "auto")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Max graph steps per request (each LLM call / tool call is a step); stops tool-call loops.
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "20"))
