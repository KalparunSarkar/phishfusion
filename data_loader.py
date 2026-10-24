"""Dataset loading, HTML fetching and a synthetic dataset generator.

The framework needs, for each sample, a URL, an optional rendered HTML string
and a binary label (1 = phishing, 0 = benign).  Records can be supplied as a CSV
or JSONL file; when HTML is absent and network access is permitted the loader
fetches it at build time, capping the response size and applying a timeout so a
single hostile host cannot stall a run.  A synthetic generator produces a
self-contained, reproduceable dataset so the whole pipeline can be exercised
(and unit-tested) without any network access.
"""

import csv
import json
import random
from pathlib import Path

from config import HTTP_MAX_BYTES, HTTP_TIMEOUT, HTTP_USER_AGENT

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:  # optional: TLS/browser impersonation to get past some bot checks
    from curl_cffi import requests as curl_requests

    HAS_CURL_CFFI = True
except ImportError:
    curl_requests = None
    HAS_CURL_CFFI = False

try:  # optional: JavaScript-challenge solver
    import cloudscraper

    HAS_CLOUDSCRAPER = True
except ImportError:
    cloudscraper = None
    HAS_CLOUDSCRAPER = False


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Markers of anti-bot / JS-challenge interstitials (not the real page).
_CHALLENGE_MARKERS = (
    "awswafcookiedomainlist", "gokuprops", "cf-chl", "challenge-platform",
    "just a moment", "enable javascript and cookies", "attention required",
    "checking your browser", "_cf_chl_opt", "ddm-form", "px-captcha",
    "bm-verify", "captcha-delivery", "geo.captcha",
)


# --------------------------------------------------------------------------- IO
def load_records(path):
    """Load records from a ``.csv`` or ``.jsonl`` file.

    Recognised keys: ``url``, ``label`` (0/1), optional ``html``.
    """
    path = Path(path)
    records = []
    if path.suffix.lower() == ".jsonl":
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(_normalise(json.loads(line)))
    else:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                records.append(_normalise(row))
    return [r for r in records if r["url"]]


def _normalise(row):
    label = row.get("label", row.get("Label", 0))
    try:
        label = int(float(label))
    except (TypeError, ValueError):
        label = 1 if str(label).strip().lower() in ("phishing", "phish", "1", "true") else 0
    return {
        "url": str(row.get("url", row.get("URL", ""))).strip(),
        "label": label,
        "html": row.get("html", row.get("HTML", "")) or "",
    }


def save_records(records, path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    else:
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["url", "label", "html"])
            writer.writeheader()
            writer.writerows(records)
    return path


# ------------------------------------------------------------------------ fetch
def _decode(resp, content):
    enc = resp.encoding
    if not enc or enc.lower() == "iso-8859-1":
        enc = getattr(resp, "apparent_encoding", None) or "utf-8"
    try:
        return content.decode(enc, errors="ignore")
    except (LookupError, TypeError):
        return content.decode("utf-8", errors="ignore")


def _is_challenge(html):
    low = html.lower()
    return any(m in low for m in _CHALLENGE_MARKERS)


def _new_client():
    """Prefer a JS-challenge solver, then TLS impersonation, then plain requests."""
    if HAS_CLOUDSCRAPER:
        return ("cloudscraper", cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}))
    if HAS_CURL_CFFI:
        return ("curl_cffi", curl_requests.Session(impersonate="chrome"))
    if requests is not None:
        s = requests.Session()
        s.headers.update(BROWSER_HEADERS)
        return ("requests", s)
    return (None, None)


def fetch_html_ex(url, timeout=HTTP_TIMEOUT, max_bytes=HTTP_MAX_BYTES, retries=2):
    """Robustly fetch a URL's HTML.

    Returns a dict: ``{html, ok, status, final_url, reason, client}``.  Handles
    missing schemes, https->http fallback, retries, browser headers, any 2xx with
    an HTML body, and detects anti-bot/JS-challenge interstitials (which are
    reported as ``ok=False`` rather than fed to the feature extractor as if they
    were the real page).
    """
    info = {"html": "", "ok": False, "status": None, "final_url": url,
            "reason": None, "client": None, "bytes": 0}
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url

    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://"):])
    elif url.startswith("http://"):
        candidates.append("https://" + url[len("http://"):])

    for candidate in candidates:
        for attempt in range(retries + 1):
            client_name, client = _new_client()
            info["client"] = client_name
            if client is None:
                info["reason"] = "no HTTP client available"
                return info
            try:
                resp = client.get(candidate, timeout=timeout, allow_redirects=True)
            except Exception as exc:
                info["reason"] = f"{type(exc).__name__}"
                continue

            info["status"] = resp.status_code
            info["final_url"] = str(getattr(resp, "url", candidate))
            raw = resp.content[:max_bytes]
            info["bytes"] = len(raw)
            if not raw:
                info["reason"] = f"empty body (HTTP {resp.status_code})"
                continue
            html = _decode(resp, raw)
            if _is_challenge(html):
                info["reason"] = "anti-bot / JS challenge page"
                continue
            if resp.status_code >= 400 and len(html) < 1000:
                info["reason"] = f"HTTP {resp.status_code}"
                continue
            info.update(html=html, ok=True, reason=None)
            return info
    return info


def fetch_html(url, timeout=HTTP_TIMEOUT, max_bytes=HTTP_MAX_BYTES):
    """Backward-compatible wrapper: returns ``(html, ok)``."""
    info = fetch_html_ex(url, timeout=timeout, max_bytes=max_bytes)
    return info["html"], info["ok"]


# ---------------------------------------------------------------- synthetic data
_BENIGN_TEMPLATE = """<!doctype html><html><head><title>{title}</title></head><body>
<h1>Welcome to {brand}</h1>
<p>Explore our products and services. Sign in to manage your account.</p>
<form action="/login" method="post">
  <input type="text" name="user">
  <input type="password" name="pass">
  <button>Sign in</button>
</form>
<a href="https://www.{brand}.com/about">About</a>
<a href="https://www.{brand}.com/contact">Contact</a>
<img src="/static/logo.png" alt="logo">
<p>&copy; {year} {brand}. All rights reserved.</p>
</body></html>"""

_PHISH_TEMPLATE = """<!doctype html><html><head><title>Verify Your Account</title>
<meta http-equiv="refresh" content="3;url=http://{evil}/next"></head><body>
<p>Dear user, your account has been limited. Verify your identity within 24 hours
by confirming your password and card details or your {brand} account will be
permanently suspended.</p>
<form action="" method="post" onsubmit="return true">
  <input type="text" name="email" placeholder="Email">
  <input type="password" name="password" placeholder="Password">
  <input type="hidden" name="token" value="x">
</form>
<a href="http://{evil}/login">Click here to verify</a>
<a href="#">Cancel</a>
<script>window.open('popup.html'); document.oncontextmenu=function(){{return false;}}</script>
<img src="http://cdn-{evil}/logo.png">
<iframe style="display:none" src="http://{evil}/frame"></iframe>
</body></html>"""

_BENIGN_BRANDS = ["google", "amazon", "wikipedia", "github", "microsoft", "apple", "netflix",
                  "linkedin", "dropbox", "adobe"]
_EVIL_HOSTS = ["evil-host", "cdn-tracker", "secure-verify", "malware-drop", "phish-kit"]
_SUS_TLDS = ["xyz", "top", "tk", "gq", "club", "click"]


def synthesize_dataset(n=400, seed=0, phishing_ratio=0.5):
    """Create a balanced, reproducible synthetic dataset of URL/HTML/label."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        is_phish = (i % 2 == 0) if i < int(n * phishing_ratio * 2) else rng.random() < phishing_ratio
        brand = rng.choice(_BENIGN_BRANDS)
        if is_phish:
            evil = rng.choice(_EVIL_HOSTS)
            tld = rng.choice(_SUS_TLDS)
            kw = rng.choice(["secure-login", "account-verify", "update-billing", "signin-confirm"])
            url = f"http://{kw}.{brand}.{evil}.{tld}/webscr?cmd=login&session={rng.randint(1000,9999)}"
            html = _PHISH_TEMPLATE.format(brand=brand, evil=evil)
        else:
            path = rng.choice(["", "search", "about", "products", "help/faq"])
            url = f"https://www.{brand}.com/{path}"
            html = _BENIGN_TEMPLATE.format(title=f"{brand.title()}", brand=brand, year=2026)
        records.append({"url": url, "label": 1 if is_phish else 0, "html": html})
    rng.shuffle(records)
    return records


if __name__ == "__main__":
    recs = synthesize_dataset(n=10, seed=1)
    for r in recs:
        print(r["label"], r["url"])
