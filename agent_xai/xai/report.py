"""Presenting LIME explanations: console table and a self-contained HTML report.

HTML report = (1) the request with every feature highlighted by its LIME weight
(red = pushes toward the explained class, blue = pushes away; diverging scale, gray = ~0),
(2) a diverging bar chart of the top features with signed values on every bar,
(3) the probe/surrogate numbers needed to judge whether to trust the explanation.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .features import RequestFeatures
from .lime import LimeExplanation

# Diverging pair (validated reference palette): red = toward class, blue = away.
CSS = """
.lime { --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --muted:#8a8984; --grid:#e4e3df;
        --pos:#e34948; --neg:#2a78d6; --mid:#f0efec; --code:#f5f4f1;
        color-scheme: light; background:var(--surface); color:var(--text);
        font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; padding:16px 20px; max-width:960px; box-sizing:border-box; }
@media (prefers-color-scheme: dark) {
  .lime { --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7; --muted:#8f8e86; --grid:#33332f;
          --pos:#e66767; --neg:#3987e5; --mid:#383835; --code:#232321; color-scheme: dark; }
}
.lime h1 { font-size:18px; margin:0 0 4px; } .lime h2 { font-size:15px; margin:24px 0 8px; }
.lime .sub { color:var(--text2); margin:0 0 12px; }
.lime .stats { display:flex; flex-wrap:wrap; gap:8px 24px; margin:8px 0; }
.lime .stat b { display:block; font-size:18px; font-variant-numeric:tabular-nums; }
.lime .stat span { color:var(--text2); font-size:12px; }
.lime .req { background:var(--code); border-radius:6px; padding:12px; overflow-x:auto;
             font:13px/1.9 ui-monospace,Consolas,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }
.lime .f { border-radius:3px; padding:1px 1px; cursor:help; }
.lime .legend { display:flex; gap:16px; align-items:center; color:var(--text2); font-size:12px; margin:6px 0; }
.lime .sw { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-2px; margin-right:4px; }
.lime .bars { display:grid; grid-template-columns:minmax(120px,32%) 1fr; gap:4px 12px; align-items:center; }
.lime .blabel { font:12px ui-monospace,Consolas,monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:right; }
.lime .track { position:relative; height:20px; overflow:hidden; }
.lime .axis { position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--grid); }
.lime .bar { position:absolute; top:3px; height:14px; }
.lime .bar.pos { left:50%; background:var(--pos); border-radius:0 4px 4px 0; }
.lime .bar.neg { right:50%; background:var(--neg); border-radius:4px 0 0 4px; }
.lime .val { position:absolute; top:0; font-size:12px; color:var(--text); font-variant-numeric:tabular-nums; white-space:nowrap; }
.lime .note { color:var(--text2); font-size:12px; margin-top:16px; }
"""


def print_explanation(e: LimeExplanation, k: int = 10) -> None:
    probs = ", ".join(f"{c}={p:.3f}" for c, p in e.original_probs.items())
    print(f"\nLIME [{e.target}/{e.granularity}] {e.request_id}  ground truth: {e.ground_truth}")
    print(f"  probe: {probs}  (candidate mass {e.candidate_mass:.2f})")
    if e.agent_choice:
        print(f"  agent actually chose: {e.agent_choice}")
    print(f"  explaining P({e.explained_class})  surrogate R^2={e.score:.2f}  "
          f"intercept={e.intercept:.3f}  local pred={e.local_pred:.3f}")
    for f in e.top(k):
        arrow = "toward" if f["weight"] > 0 else "away from"
        print(f"   {f['weight']:+.4f}  {arrow:9s} {e.explained_class:14s} | {f['label']}")


def _color(w: float, wmax: float) -> str:
    a = min(abs(w) / wmax, 1.0) if wmax else 0.0
    var = "--pos" if w > 0 else "--neg"
    return f"color-mix(in srgb, var({var}) {a * 70:.0f}%, var(--mid))"


def _highlight(text: str, spans: list[tuple[tuple[int, int], float, str]], wmax: float) -> str:
    out, pos = [], 0
    for (a, b), w, label in sorted(spans):
        out.append(html.escape(text[pos:a]))
        tip = html.escape(f"{label}  weight {w:+.4f}")
        out.append(f'<span class="f" style="background:{_color(w, wmax)}" title="{tip}">'
                   f"{html.escape(text[a:b])}</span>")
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _request_view(e: LimeExplanation, rf: RequestFeatures, wmax: float) -> str:
    by_target: dict[tuple, list] = {}
    for f in e.features:
        by_target.setdefault((f["part"], f["param_idx"]), []).append(f)

    def param_line(part: str) -> str:
        pieces = []
        for i, p in enumerate(rf.parts[part]):
            fs = by_target.get((part, i), [])
            if e.granularity == "param" and fs:
                f = fs[0]
                shown = f"{p.name}={p.value}" if p.raw_value is not None else p.name
                pieces.append(_highlight(shown, [((0, len(shown)), f["weight"], f["label"])], wmax))
            else:
                spans = [(tuple(f["span"]), f["weight"], f["label"]) for f in fs]
                val = _highlight(p.value, spans, wmax)
                pieces.append(html.escape(p.name) + ("=" + val if p.raw_value is not None else ""))
        return "&amp;".join(pieces)

    path_spans = [(tuple(f["span"]), f["weight"], f["label"]) for f in by_target.get(("path", -1), [])]
    line = f"{html.escape(rf.req.method)} {html.escape(rf.base)}{_highlight(rf.path, path_spans, wmax)}"
    if rf.parts["query"]:
        line += "?" + param_line("query")
    body = param_line("body")
    return line + ("\n\n" + body if body else "")


def _bars(e: LimeExplanation, k: int, wmax: float) -> str:
    rows = []
    for f in e.top(k):
        w = f["weight"]
        pct = 40 * abs(w) / wmax if wmax else 0  # leave room for the value label
        side = "pos" if w > 0 else "neg"
        val_pos = f"left:calc(50% + {pct:.1f}% + 4px)" if w > 0 else f"right:calc(50% + {pct:.1f}% + 4px)"
        tip = html.escape(f"{f['label']}: {w:+.4f}")
        rows.append(
            f'<div class="blabel" title="{html.escape(f["label"])}">{html.escape(f["label"])}</div>'
            f'<div class="track" title="{tip}"><div class="axis"></div>'
            f'<div class="bar {side}" style="width:{pct:.1f}%"></div>'
            f'<div class="val" style="{val_pos}">{w:+.3f}</div></div>')
    return '<div class="bars">' + "".join(rows) + "</div>"


def to_html(e: LimeExplanation, req, k: int = 15) -> str:
    rf = RequestFeatures(req, e.granularity)
    wmax = max((abs(f["weight"]) for f in e.features), default=0) or 1.0
    probs = "".join(
        f'<div class="stat"><b>{p:.3f}</b><span>P({html.escape(c)})</span></div>'
        for c, p in e.original_probs.items())
    agent = (f'<div class="stat"><b>{html.escape(e.agent_choice)}</b><span>agent chose</span></div>'
             if e.agent_choice else "")
    trust = "good" if e.score >= 0.5 else "weak - read with care"
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIME {html.escape(e.request_id)}</title><style>{CSS}</style></head><body style="margin:0;background:#fcfcfb">
<style>@media (prefers-color-scheme: dark){{body{{background:#1a1a19!important}}}}</style>
<div class="lime">
<h1>LIME explanation &middot; {html.escape(e.request_id)}</h1>
<p class="sub">Target <b>{e.target}</b>, features = <b>{e.granularity}s</b>, {e.num_samples} perturbed samples.
Explaining <b>P({html.escape(e.explained_class)})</b>. Ground truth: <b>{html.escape(e.ground_truth)}</b>.</p>
<div class="stats">{probs}{agent}
<div class="stat"><b>{e.score:.2f}</b><span>surrogate R&sup2; ({trust})</span></div>
<div class="stat"><b>{e.candidate_mass:.2f}</b><span>candidate mass</span></div></div>

<h2>Request, highlighted by LIME weight</h2>
<div class="legend"><span><span class="sw" style="background:var(--pos)"></span>raises P({html.escape(e.explained_class)})</span>
<span><span class="sw" style="background:var(--neg)"></span>lowers it</span><span>hover a token for its weight</span></div>
<div class="req">{_request_view(e, rf, wmax)}</div>

<h2>Top {min(k, len(e.features))} features by |weight|</h2>
{_bars(e, k, wmax)}
<p class="note">Weight = change in P({html.escape(e.explained_class)}) predicted by the local linear surrogate when the
feature is kept vs. removed. Intercept {e.intercept:.3f}; surrogate prediction for the original request
{e.local_pred:.3f} vs. probe {e.original_probs[e.explained_class]:.3f}. Headers, method and host are held fixed.</p>
</div></body></html>"""


def save_explanation(e: LimeExplanation, req, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{e.request_id.replace(':', '_')}_{e.target}_{e.granularity}"
    (out_dir / f"{stem}.json").write_text(json.dumps(e.to_dict(), indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    path = out_dir / f"{stem}.html"
    path.write_text(to_html(e, req), encoding="utf-8")
    return path
