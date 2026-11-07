"""Parser for the raw CSIC 2010 text files.

Each file is a sequence of raw HTTP requests. A request starts with a
request line ("GET http://... HTTP/1.1"), followed by headers, a blank line,
and (for POST/PUT) a URL-encoded body.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from urllib.parse import unquote_plus

from .config import DATASETS

_REQUEST_LINE = re.compile(r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS) (\S+) HTTP/\d\.\d$")


@dataclass
class HttpRequest:
    dataset: str
    index: int
    label: str  # ground truth: "normal" | "anomalous" (never shown to the agent)
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def request_id(self) -> str:
        return f"{self.dataset}:{self.index}"

    @property
    def decoded_url(self) -> str:
        return unquote_plus(self.url, encoding="latin-1")

    @property
    def decoded_body(self) -> str:
        return unquote_plus(self.body, encoding="latin-1")

    def to_raw(self) -> str:
        """Re-assemble the request as raw HTTP text (what the agent sees)."""
        lines = [f"{self.method} {self.url} HTTP/1.1"]
        lines += [f"{k}: {v}" for k, v in self.headers.items()]
        raw = "\n".join(lines) + "\n"
        if self.body:
            raw += "\n" + self.body + "\n"
        return raw

    def to_dict(self) -> dict:
        d = asdict(self)
        d["request_id"] = self.request_id
        return d


def _parse_file(dataset: str) -> list[HttpRequest]:
    path, label = DATASETS[dataset]
    text = path.read_text(encoding="latin-1")

    requests: list[HttpRequest] = []
    current: HttpRequest | None = None
    in_body = False

    for line in text.splitlines():
        m = _REQUEST_LINE.match(line)
        if m:
            if current:
                requests.append(current)
            current = HttpRequest(dataset, len(requests), label, m.group(1), m.group(2))
            in_body = False
            continue
        if current is None:
            continue
        if not line.strip():
            in_body = True  # blank line ends headers
            continue
        if in_body:
            current.body += line
        elif ":" in line:
            k, v = line.split(":", 1)
            current.headers[k.strip()] = v.strip()

    if current:
        requests.append(current)
    return requests


@lru_cache(maxsize=None)
def load_dataset(dataset: str) -> tuple[HttpRequest, ...]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    return tuple(_parse_file(dataset))


def get_request(dataset: str, index: int) -> HttpRequest:
    return load_dataset(dataset)[index]


def sample_requests(n: int, seed: int = 42, datasets: list[str] | None = None) -> list[HttpRequest]:
    """Random, label-balanced sample across the given datasets."""
    datasets = datasets or ["normal_test", "anomalous_test"]
    rng = random.Random(seed)
    per_ds = max(1, n // len(datasets))
    picked: list[HttpRequest] = []
    for ds in datasets:
        picked += rng.sample(load_dataset(ds), per_ds)
    rng.shuffle(picked)
    return picked[:n]
