"""URL lexical and statistical feature extractor.

Extends the base-paper feature philosophy (Aljofey et al., 2022) with two
important corrections over the previous version:

* hostname decomposition uses ``tldextract`` (with an offline fallback) so the
  *registered* domain - not an arbitrary label - is used for brand comparison;
* feature vectors are emitted through a canonical key list so that training-time
  and inference-time ordering can never diverge.
"""

import math
import re
import socket
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse

from config import (
    POPULAR_BRANDS,
    SHORTENERS,
    SUSPICIOUS_TLDS,
    SUSPICIOUS_WORDS,
    URL_NETWORK_KEYS,
    URL_NUMERIC_KEYS,
)

try:  # optional
    import whois
except ImportError:  # pragma: no cover - optional dependency
    whois = None

try:  # optional C-accelerated edit distance
    import Levenshtein

    def edit_distance(a, b):
        return Levenshtein.distance(a, b)
except ImportError:  # pure-python fallback
    def edit_distance(a, b):
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = curr
        return prev[-1]

try:  # offline-friendly public-suffix parsing
    import tldextract

    _TLD = tldextract.TLDExtract(suffix_list_urls=())
except Exception:  # pragma: no cover
    _TLD = None


def split_host(host):
    """Return (subdomain, domain, suffix) for a hostname.

    Falls back to naive dot-splitting when the public-suffix library is absent.
    """
    host = (host or "").strip().lower()
    if not host:
        return "", "", ""
    if _TLD is not None:
        ext = _TLD(host)
        return ext.subdomain, ext.domain, ext.suffix
    parts = host.split(".")
    if len(parts) == 1:
        return "", parts[0], ""
    if len(parts) == 2:
        return "", parts[0], parts[1]
    return ".".join(parts[:-2]), parts[-2], parts[-1]


def registered_domain(host):
    """Second-level / registrable domain used for brand comparisons."""
    _sub, domain, suffix = split_host(host)
    if not domain:
        return (host or "").lower()
    return f"{domain}.{suffix}" if suffix else domain


def shannon_entropy(s):
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def has_ip_address(host):
    pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if not re.match(pattern, host or ""):
        return 0
    # reject malformed octets such as 999.1.1.1
    return int(all(0 <= int(o) <= 255 for o in host.split(".")))


def has_punycode(host):
    return int("xn--" in (host or "").lower())


def get_tld(host):
    _sub, _domain, suffix = split_host(host)
    if suffix:
        return suffix.lower()
    parts = (host or "").split(".")
    return parts[-1].lower() if len(parts) > 1 else ""


def min_brand_edit_distance(host):
    """Minimum edit distance between the registered domain and a known brand."""
    domain = split_host(host)[1]
    if not domain:
        return -1
    return min(edit_distance(domain.lower(), b) for b in POPULAR_BRANDS)


def is_shortened(host):
    host = (host or "").lower()
    return int(any(host == s or host.endswith("." + s) for s in SHORTENERS))


def has_dns_record(host):
    try:
        socket.gethostbyname(host)
        return 1
    except Exception:
        return 0


def domain_age_days(host):
    if whois is None:
        return -1
    try:
        info = whois.whois(host)
        created = info.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return -1
        return (datetime.now() - created).days
    except Exception:
        return -1


def _safe_port(parsed):
    try:
        return 1 if parsed.port else 0
    except ValueError:
        return 1  # malformed/ambiguous port is itself suspicious


def extract_url_features(url, use_network_features=False):
    """Extract a dict of lexical/statistical URL features.

    The dict always contains every key in ``URL_NUMERIC_KEYS`` plus ``tld`` and,
    when requested, ``URL_NETWORK_KEYS``.
    """
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split(":")[0].lower()
    path = parsed.path
    query = parsed.query
    full = url

    features = {}
    features["url_length"] = len(full)
    features["host_length"] = len(host)
    features["path_length"] = len(path)
    features["num_dots"] = full.count(".")
    features["num_hyphens"] = full.count("-")
    features["num_underscores"] = full.count("_")
    features["num_slashes"] = full.count("/")
    features["num_digits"] = sum(c.isdigit() for c in full)
    features["num_at"] = full.count("@")
    features["num_question"] = full.count("?")
    features["num_equal"] = full.count("=")
    features["num_percent"] = full.count("%")
    features["num_ampersand"] = full.count("&")
    features["num_dollar"] = full.count("$")
    features["num_exclamation"] = full.count("!")
    features["num_hash"] = full.count("#")
    features["num_subdomains"] = max(host.count(".") - 1, 0)
    features["has_ip"] = has_ip_address(host)
    features["has_https"] = 1 if parsed.scheme == "https" else 0
    features["has_port"] = _safe_port(parsed)
    features["is_shortened"] = is_shortened(host)
    features["suspicious_word_count"] = sum(w in full.lower() for w in SUSPICIOUS_WORDS)
    features["digit_ratio"] = features["num_digits"] / len(full) if full else 0
    features["url_entropy"] = shannon_entropy(full)
    features["host_entropy"] = shannon_entropy(host)
    features["query_length"] = len(query)
    features["has_query"] = int(bool(query))
    features["is_suspicious_tld"] = int(get_tld(host) in SUSPICIOUS_TLDS)
    features["has_punycode"] = has_punycode(host)
    features["has_double_slash_redirect"] = int("//" in path)
    features["brand_edit_distance"] = min_brand_edit_distance(host)
    features["ends_with_known_ext"] = int(
        bool(re.search(r"\.(exe|zip|php|html?|scr|apk)$", path.lower()))
    )
    features["longest_word_length"] = max(
        (len(w) for w in re.split(r"[./?=&_\-]", full) if w), default=0
    )
    features["tld"] = get_tld(host)

    if use_network_features:
        features["has_dns_record"] = has_dns_record(host)
        features["domain_age_days"] = domain_age_days(host)
    return features


def url_feature_vector(features, include_network=False):
    """Flatten a feature dict into a canonical-order numeric vector."""
    keys = URL_NUMERIC_KEYS + (URL_NETWORK_KEYS if include_network else [])
    return [float(features.get(k, 0.0)) for k in keys]


def extract_char_ngrams(url, n=3):
    url = url.lower()
    return [url[i:i + n] for i in range(len(url) - n + 1)]


if __name__ == "__main__":
    sample = "http://secure-login.paypa1.com.verify-account.xyz/webscr?cmd=login"
    feats = extract_url_features(sample)
    print(f"registered_domain={registered_domain(urlparse(sample).netloc)}")
    for k, v in feats.items():
        print(f"  {k:28} {v}")
    print("vector len:", len(url_feature_vector(feats)))
    print("ngrams:", extract_char_ngrams(sample)[:10])
