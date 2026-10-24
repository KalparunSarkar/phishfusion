"""Interactive demonstration of the URL-HTML-LLM phishing detector.

Two modes, both driven by the trained model in ``artifacts/phish_model`` and
real held-out samples from ``data/phreshphish_test.csv`` (so nothing depends on
live phishing URLs, which are usually already down):

* terminal walkthrough (default):  ``python demo.py``
* local web UI (recommended for a class demo):  ``python demo.py --web``

Flags: ``--fast`` (no pauses), ``--llm`` (add LLM-generated explanations),
``--port``/``--host`` for the web server.
"""

import argparse
import html as _html
import sys
import time

from config import DATA_DIR
from data_loader import load_records, synthesize_dataset
from pipeline import PhishingPipeline

# --------------------------------------------------------------------------- colour
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    enabled = True

    @classmethod
    def w(cls, text, colour):
        return f"{colour}{text}{cls.RESET}" if cls.enabled else text

    @classmethod
    def disable(cls):
        cls.enabled = False
        for name in ("RESET", "BOLD", "DIM", "RED", "GREEN", "YELLOW", "BLUE", "CYAN"):
            setattr(cls, name, "")


def section(title):
    line = "=" * 74
    print(f"\n{C.w(line, C.CYAN)}\n{C.BOLD}{title}{C.RESET}\n{C.w(line, C.CYAN)}")


def pause(args, msg="  [press Enter]"):
    if args.fast or not sys.stdin.isatty():
        return
    try:
        input(C.w(msg, C.DIM))
    except EOFError:
        pass


# --------------------------------------------------------------------------- setup
def load_pipeline():
    try:
        pipe = PhishingPipeline.load("phish_model")
        print(C.w("  loaded trained model from artifacts/phish_model", C.GREEN))
        return pipe
    except Exception as exc:
        print(C.w(f"  no saved model ({exc}); training a quick one...", C.YELLOW))
        records = synthesize_dataset(n=400, seed=0)
        pipe = PhishingPipeline(seed=0)
        pipe.fit(records, mode="attention", head="xgb", use_llm=False, verbose=False)
        pipe.save()
        return pipe


def _test_records():
    path = DATA_DIR / "phreshphish_test.csv"
    if path.exists():
        return load_records(path), "PhreshPhish (real, held-out)"
    return synthesize_dataset(n=200, seed=1), "synthetic"


def gather_examples(pipe, n_each=3, scan=80):
    """Pick correctly-classified phishing/benign samples with modest page size."""
    records, _ = _test_records()
    phish = sorted([r for r in records if r["label"] == 1], key=lambda r: len(r["html"]))
    benign = sorted([r for r in records if r["label"] == 0], key=lambda r: len(r["html"]))

    def pick(cands):
        chosen = []
        for r in cands[:scan]:
            res = pipe.predict(r["url"], html=r["html"], fetch=False, explain=False,
                               use_llm=False)
            if res["verdict"] == r["label"]:
                chosen.append(r)
            if len(chosen) >= n_each:
                break
        return chosen or cands[:n_each]

    return pick(phish) + pick(benign)


def show_prediction(pipe, rec, use_llm=False, explain=True):
    res = pipe.predict(rec["url"], html=rec["html"], fetch=False, explain=explain,
                       use_llm=use_llm)
    truth = "phishing" if rec["label"] == 1 else "legitimate"
    correct = res["verdict"] == rec["label"]
    verdict_colour = C.RED if res["verdict"] == 1 else C.GREEN
    print(f"\n  {C.BOLD}URL:{C.RESET} {res['url'][:96]}")
    print(f"  {C.BOLD}Truth:{C.RESET} {truth}   "
          f"{C.BOLD}Prediction:{C.RESET} "
          f"{C.w(res['label'].upper(), verdict_colour)} "
          f"({res['phishing_probability'] * 100:.1f}% phishing)  "
          f"{C.w('✓', C.GREEN) if correct else C.w('✗', C.RED)}")
    if res.get("modality_weights"):
        w = res["modality_weights"]
        print(f"  {C.BOLD}Attention weights:{C.RESET} "
              f"url={w['url']:.2f}  html={w['html']:.2f}  llm={w['llm']:.2f}")
    if res.get("top_features"):
        tops = ", ".join(f"{f['feature']} ({f['contribution']:+.3f})"
                         for f in res["top_features"][:5])
        print(f"  {C.BOLD}Top features:{C.RESET} {tops}")
    if res.get("explanation"):
        print(f"  {C.BOLD}Explanation:{C.RESET} {res['explanation']}")
    return res


# ---------------------------------------------------------------------- terminal
def run_cli(args):
    if args.no_color or not sys.stdout.isatty():
        C.disable()
    print(C.w("\n  URL–HTML–LLM Fusion Framework — Phishing Detection Demo", C.BOLD))

    section("1. Model & pipeline")
    pipe = load_pipeline()
    print(f"  Fusion: {C.w(pipe.model.fusion_mode, C.BLUE)}   "
          f"Head: {C.w(pipe.model.head, C.BLUE)}")
    u, h, l = pipe.dims
    print(f"  Feature blocks: URL={u}  HTML={h}  LLM semantic={l}  "
          f"(total {u + h + l})")
    print("  Stages: URL features -> HTML/DOM features -> LLM semantic signals\n"
          "          -> learned attention fusion -> classifier -> SHAP explanation")
    pause(args)

    section("2. Live evaluation on held-out real pages")
    records, source = _test_records()
    n = min(300, len(records))
    t0 = time.time()
    metrics = pipe.evaluate_records(records, use_llm=False, limit=n)
    print(f"  Dataset: {source}  (evaluating {metrics['n']} samples in "
          f"{time.time() - t0:.1f}s)")
    print(f"  Accuracy : {C.w(f'{metrics['accuracy']:.3f}', C.GREEN)}")
    print(f"  F1-score : {metrics['f1']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"  FPR      : {metrics['fpr']:.3f}   "
          f"(TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']})")
    pause(args)

    section("3. Case-by-case predictions (real held-out pages)")
    for rec in gather_examples(pipe, n_each=3):
        show_prediction(pipe, rec, use_llm=args.llm)
        pause(args)

    section("4. Try your own URL")
    if args.url:
        urls = [args.url]
    elif args.fast or not sys.stdin.isatty():
        urls = []
        print("  (non-interactive: pass --url to classify a specific link)")
    else:
        urls = []
        while True:
            u = input(C.w("  Enter a URL (blank to finish): ", C.DIM)).strip()
            if not u:
                break
            urls.append(u)
    for u in urls:
        print(f"\n  {C.BOLD}URL:{C.RESET} {u}")
        res = pipe.predict(u, fetch=True, explain=True, use_llm=args.llm)
        colour = C.RED if res["verdict"] == 1 else C.GREEN
        fetch = res.get("fetch") or {}
        fetched = "yes" if res["html_available"] else \
                  f"no ({fetch.get('reason', 'unreachable')})"
        print(f"  Verdict: {C.w(res['label'].upper(), colour)} "
              f"({res['phishing_probability'] * 100:.1f}%)  HTML fetched: {fetched}")
        print(f"  Explanation: {res.get('explanation', '')}")

    section("Done")
    print("  Web demo:  python demo.py --web   ->  http://127.0.0.1:5000")
    print("  Evaluate:  python summarize_results.py  ->  artifacts/RESULTS.md\n")


# --------------------------------------------------------------------------- web
_CSS = """
:root{--bg:#ffffff;--card:#ffffff;--fg:#000000;--dim:#6b6b6b;--line:#000000;
--track:#e6e6e6;--ac:#000000;--ph:#000000;--ok:#000000;--pos:#000000;--neg:#a3a3a3;}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--fg);min-height:100vh}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 60px}
h1{font-size:26px;margin:0 0 4px}.sub{color:var(--dim);margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:18px;
box-shadow:0 6px 24px rgba(0,0,0,.10)}
label{display:block;font-size:13px;color:var(--dim);margin:12px 0 6px}
input[type=text],textarea{width:100%;background:#fff;border:1px solid var(--line);color:var(--fg);
border-radius:10px;padding:12px;font-size:15px;font-family:inherit}
textarea{height:80px;resize:vertical}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:16px}
button{background:var(--ac);color:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 20px;
font-size:15px;font-weight:700;cursor:pointer}button:hover{background:#333}
button.ghost{background:#fff;color:var(--fg);border:1px solid var(--line);font-weight:600}
button.ghost:hover{background:#000;color:#fff}
.badge{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:800;letter-spacing:.3px;
border:1px solid var(--line)}
.badge.ph{background:#000;color:#fff}
.badge.ok{background:#fff;color:#000}
.big{font-size:40px;font-weight:800;margin:8px 0 2px;color:#000}
.bar{height:12px;background:var(--track);border-radius:8px;overflow:hidden;margin:6px 0 14px;
border:1px solid #d0d0d0}
.bar>span{display:block;height:100%;border-radius:8px;background:#000}
.wrow{display:grid;grid-template-columns:60px 1fr 56px;gap:10px;align-items:center;font-size:14px;margin:6px 0}
.feat{display:grid;grid-template-columns:1fr 160px;gap:10px;align-items:center;font-size:13px;margin:4px 0}
.fbar{height:9px;background:var(--track);border-radius:6px;overflow:hidden;border:1px solid #d0d0d0}
.fbar>span{display:block;height:100%}
.ex{border:1px dashed var(--line);border-radius:12px;padding:12px;margin-top:10px;font-size:13px;color:var(--dim)}
.ex b{color:var(--fg)}
pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid var(--line);border-radius:10px;
padding:12px;font-size:13px;color:#000}
.muted{color:var(--dim);font-size:12px;margin-top:8px}
"""

_WEB_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phishing Detection Demo</title><style>%s</style></head><body><div class="wrap">""" % _CSS


def _result_html(res):
    ph = res["verdict"] == 1
    colour = "var(--ph)" if ph else "var(--ok)"
    badge = f'<span class="badge {"ph" if ph else "ok"}">{res["label"].upper()}</span>'
    proba = res["phishing_probability"] * 100
    fetch = res.get("fetch") or {}
    parts = [f'<div class="card"><div>{badge}</div>'
             f'<div class="big" style="color:{colour}">{proba:.1f}%</div>'
             f'<div class="sub" style="margin:0">phishing probability</div>'
             f'<div class="bar"><span style="width:{proba:.1f}%;background:{colour}"></span></div>']
    if fetch and not res["html_available"]:
        status = f" (HTTP {fetch['status']})" if fetch.get("status") else ""
        reason = _html.escape(str(fetch.get("reason") or "unreachable"))
        parts.append(f'<div class="muted" style="color:#000;font-weight:600">Could not fetch page HTML '
                     f'({reason}){status} — verdict is based on the URL only, which is less '
                     f'reliable.</div>')
    meta = f'URL: {_html.escape(res["url"])} &nbsp;|&nbsp; HTML fetched: {res["html_available"]}'
    if fetch.get("client"):
        meta += f' &nbsp;|&nbsp; client: {fetch["client"]}'
    parts.append(f'<div class="muted">{meta}</div>')
    if res.get("modality_weights"):
        parts.append('<div style="margin-top:16px"><label>Modality weights '
                     '(learned attention)</label>')
        for k, v in res["modality_weights"].items():
            parts.append(f'<div class="wrow"><span>{k}</span>'
                         f'<div class="bar" style="margin:0"><span style="width:{v*100:.0f}%;'
                         f'background:var(--ac)"></span></div><span>{v:.2f}</span></div>')
        parts.append('</div>')
    if res.get("top_features"):
        parts.append('<div style="margin-top:16px"><label>Top contributing features '
                     '(SHAP)</label>')
        mx = max(abs(f["contribution"]) for f in res["top_features"]) or 1e-9
        for f in res["top_features"]:
            c = f["contribution"]
            w = abs(c) / mx * 100
            col = "var(--pos)" if c > 0 else "var(--neg)"
            parts.append(f'<div class="feat"><span>{_html.escape(f["feature"])} '
                         f'({c:+.3f})</span><div class="fbar"><span style="width:{w:.0f}%;'
                         f'background:{col}"></span></div></div>')
        parts.append('</div>')
    if res.get("explanation"):
        parts.append(f'<div style="margin-top:16px"><label>Explanation</label>'
                     f'<pre>{_html.escape(res["explanation"])}</pre></div>')
    parts.append('</div>')
    return "".join(parts)


def run_web(args):
    from flask import Flask, request

    pipe = PhishingPipeline.load("phish_model")
    records, _ = _test_records()
    examples = gather_examples(pipe, n_each=2)
    EX = {str(i): r for i, r in enumerate(examples)}

    app = Flask("demo")
    app.pipe = pipe

    def page(result=""):
        ex_html = "".join(
            f'<div class="ex"><b>{"PHISHING" if r["label"] else "LEGITIMATE"}</b> — '
            f'{_html.escape(r["url"][:80])} '
            f'<form method="post" action="/predict" style="display:inline">'
            f'<input type="hidden" name="ex" value="{i}">'
            f'<button class="ghost" style="padding:4px 10px;margin-top:8px">Try it</button>'
            f'</form></div>'
            for i, r in EX.items())
        return _WEB_HEAD + """
<div class="card">
  <h1>URL–HTML–LLM Phishing Detector</h1>
  <div class="sub">Tri-modal fusion: URL features + HTML/DOM features + LLM semantic
  reasoning, with a learned attention layer and SHAP-based explanations.</div>
  <form method="post" action="/predict">
    <label>URL to analyse</label>
    <input type="text" name="url" placeholder="https://example.com/"
           value="example.com">
    <label>Optional: paste page HTML (leave blank to fetch live)</label>
    <textarea name="html" placeholder="<html>... optional ...</html>"></textarea>
    <div class="row">
      <button type="submit">Analyse URL</button>
      <label style="margin:0"><input type="checkbox" name="llm" value="1"> add LLM explanation</label>
    </div>
  </form>
  <div style="margin-top:18px"><label>Or try a real held-out example</label>""" \
            + ex_html + """
</div>""" + result + "</div></body></html>"

    @app.route("/", methods=["GET"])
    def index():
        return page()

    @app.route("/predict", methods=["POST"])
    def predict_route():
        ex = request.form.get("ex")
        use_llm = request.form.get("llm") == "1"
        if ex and ex in EX:
            rec = EX[ex]
            res = pipe.predict(rec["url"], html=rec["html"], fetch=False, explain=True,
                               use_llm=use_llm)
        else:
            url = (request.form.get("url") or "").strip()
            html = request.form.get("html") or None
            if not url:
                return page('<div class="card"><b>Please enter a URL.</b></div>')
            res = pipe.predict(url, html=html, fetch=True, explain=True, use_llm=use_llm)
        return page(_result_html(res))

    print(f"\n  Demo server running at  http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    app.run(host=args.host, port=args.port, debug=False)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Phishing detector demo")
    ap.add_argument("--web", action="store_true", help="launch the web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--llm", action="store_true", help="use the LLM for explanations")
    ap.add_argument("--fast", action="store_true", help="no pauses / non-interactive")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--url", default=None, help="classify this URL at the end")
    args = ap.parse_args()
    if args.web:
        run_web(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
