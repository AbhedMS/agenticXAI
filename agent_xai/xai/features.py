"""Interpretable features for LIME on an HTTP request.

LIME needs features a human can read, which it switches on/off. We use the parts of a
request that carry user input:

  granularity="token": every word / symbol-run inside the URL path and inside each
                       query/body parameter value (e.g. "'", "DROP", "TABLE", "<", "script").
                       Parameter names are kept fixed.
  granularity="param": every query/body parameter ("cantidad=...") as one feature.

Headers, method, scheme and host are kept fixed (in CSIC 2010 they barely vary).
Switching a feature off *deletes* it; the request is then re-encoded exactly like the
original (URL-encoding, Content-Length), so the model always sees a well-formed request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from urllib.parse import quote_plus, unquote_plus, urlsplit

import numpy as np

from ..data_loader import HttpRequest

TOKEN_RE = re.compile(r"\w+|[^\w\s]+")  # words, or runs of symbols like "';" "--" "<"


def _dec(s: str) -> str:
    return unquote_plus(s, encoding="latin-1")


def _enc(s: str) -> str:
    return quote_plus(s, safe="*", encoding="latin-1", errors="replace")


@dataclass
class Param:
    raw_name: str
    raw_value: str | None  # None when the pair has no "="

    @property
    def name(self) -> str:
        return _dec(self.raw_name)

    @property
    def value(self) -> str:
        return _dec(self.raw_value or "")


def _parse_params(s: str) -> list[Param]:
    out = []
    for pair in filter(None, s.split("&")):
        name, eq, value = pair.partition("=")
        out.append(Param(name, value if eq else None))
    return out


@dataclass
class Feature:
    part: str          # "path" | "query" | "body"
    param_idx: int     # index into that part's params (-1 for path)
    span: tuple[int, int] | None  # char span inside the decoded value/path; None = whole param
    text: str          # what the feature looks like (for display)
    param_name: str = ""

    @property
    def label(self) -> str:
        where = "path" if self.part == "path" else f"{self.part}.{self.param_name}"
        return f"{where}: {self.text}"


class RequestFeatures:
    """Features of one request + rebuilding the request for any on/off mask."""

    def __init__(self, req: HttpRequest, granularity: str = "token"):
        if granularity not in ("token", "param"):
            raise ValueError("granularity must be 'token' or 'param'")
        self.req = req
        self.granularity = granularity

        u = urlsplit(req.url)
        self.base = f"{u.scheme}://{u.netloc}" if u.scheme else ""
        self.raw_path = u.path
        self.path = _dec(u.path)
        self.parts = {"query": _parse_params(u.query), "body": _parse_params(req.body)}

        self.features: list[Feature] = []
        if granularity == "token":
            for m in TOKEN_RE.finditer(self.path):
                if m.group() != "/":  # keep path separators fixed
                    self.features.append(Feature("path", -1, m.span(), m.group()))
        for part, params in self.parts.items():
            for i, p in enumerate(params):
                if granularity == "param":
                    shown = f"{p.name}={p.value}" if p.raw_value is not None else p.name
                    self.features.append(Feature(part, i, None, shown, p.name))
                else:
                    for m in TOKEN_RE.finditer(p.value):
                        self.features.append(Feature(part, i, m.span(), m.group(), p.name))

    def __len__(self) -> int:
        return len(self.features)

    # -- rebuilding -------------------------------------------------------------------

    @staticmethod
    def _delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
        for a, b in sorted(spans, reverse=True):
            text = text[:a] + text[b:]
        return text

    def rebuild(self, mask: np.ndarray) -> HttpRequest:
        """Return a copy of the request with features where mask==0 removed."""
        off = [f for f, keep in zip(self.features, mask) if not keep]

        # path
        path_spans = [f.span for f in off if f.part == "path"]
        path = self.raw_path if not path_spans else \
            "/".join(_enc(seg) for seg in self._delete_spans(self.path, path_spans).split("/"))

        # params
        encoded = {}
        for part, params in self.parts.items():
            pieces = []
            for i, p in enumerate(params):
                mine = [f for f in off if f.part == part and f.param_idx == i]
                if not mine:
                    pieces.append(p.raw_name + ("" if p.raw_value is None else "=" + p.raw_value))
                elif mine[0].span is None:  # param granularity: drop whole parameter
                    continue
                else:
                    value = self._delete_spans(p.value, [f.span for f in mine])
                    pieces.append(f"{p.raw_name}={_enc(value)}")
            encoded[part] = "&".join(pieces)

        url = self.base + path + (f"?{encoded['query']}" if encoded["query"] else "")
        headers = dict(self.req.headers)
        if "Content-Length" in headers:
            headers["Content-Length"] = str(len(encoded["body"]))
        return replace(self.req, url=url, headers=headers, body=encoded["body"])
