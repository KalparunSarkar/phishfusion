"""Aggregate experiment JSON files into a single Markdown report.

Reads every ``artifacts/*ablation*.json`` produced by ``run_experiments.py`` and
writes ``artifacts/RESULTS.md`` with one table per configuration (fusion mode,
head, LLM on/off) plus the dataset used.
"""

import json
from pathlib import Path

from config import ARTIFACT_DIR

ORDER = [
    "url", "html", "llm",
    "url+html (base paper)",
    "url+llm", "html+llm",
    "url+html+llm (proposed)",
]


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def build_report(out_path=None):
    out_path = Path(out_path) if out_path else ARTIFACT_DIR / "RESULTS.md"
    files = sorted(ARTIFACT_DIR.glob("*ablation*.json"))
    lines = ["# Experiment Results", "",
             "Metrics on the **PhreshPhish** dataset (raw URL + raw HTML).",
             "Each block is a full modality ablation.", ""]
    for f in files:
        data = json.loads(f.read_text())
        cfg = data.get("config", {})
        if cfg.get("dataset") == "synthetic" or f.name == "ablation.json":
            continue  # skip synthetic smoke-test output
        results = data.get("results", {})
        lines.append(f"## `{f.name}`")
        lines.append("")
        lines.append(f"- fusion: **{cfg.get('mode','?')}**, head: **{cfg.get('head','?')}**, "
                     f"LLM features: **{cfg.get('llm', False)}**")
        lines.append(f"- dataset: `{cfg.get('dataset','?')}` "
                     f"(train={cfg.get('n_train','?')}, test={cfg.get('n_test','?')})")
        lines.append("")
        lines.append("| configuration | acc | precision | recall | F1 | ROC-AUC | FPR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name in ORDER:
            if name not in results:
                continue
            r = results[name]
            lines.append(f"| {name} | {_fmt(r['accuracy'])} | {_fmt(r['precision'])} | "
                         f"{_fmt(r['recall'])} | {_fmt(r['f1'])} | {_fmt(r['roc_auc'])} | "
                         f"{_fmt(r['fpr'])} |")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    build_report()
