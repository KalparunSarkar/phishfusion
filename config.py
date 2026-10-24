"""Central configuration for the URL-HTML-LLM phishing detection framework.

All constants that control feature extraction, fusion and I/O live here so that
the feature ordering used at training time is guaranteed to match inference time.
"""

import os
from pathlib import Path

# On macOS, PyTorch and XGBoost each bundle their own OpenMP runtime; loading both
# in one process can segfault.  Pinning OpenMP to a single thread before either
# library is imported avoids the duplicate-runtime crash.  ``setdefault`` lets a
# user override it, at their own risk.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Ollama / LLM configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5-coder:7b"
LLM_TIMEOUT = 60
LLM_MAX_TEXT_CHARS = 4000
LLM_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Lexical / heuristic vocabularies
# ---------------------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "confirm", "signin", "password", "ebayisapi", "webscr",
    "recovery", "billing", "alert", "suspend", "unlock",
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
]

SUSPICIOUS_TLDS = [
    "xyz", "top", "club", "gq", "tk", "ml", "cf", "ga", "info",
    "work", "click", "link", "country", "stream", "download",
]

POPULAR_BRANDS = [
    "paypal", "google", "facebook", "microsoft", "apple", "amazon",
    "netflix", "instagram", "bankofamerica", "chase", "wellsfargo",
    "dropbox", "linkedin", "twitter", "outlook", "yahoo", "adobe",
]

# ---------------------------------------------------------------------------
# URL feature schema (order is canonical and must not change casually)
# ---------------------------------------------------------------------------
URL_NUMERIC_KEYS = [
    "url_length", "host_length", "path_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_digits", "num_at", "num_question", "num_equal", "num_percent",
    "num_ampersand", "num_dollar", "num_exclamation", "num_hash",
    "num_subdomains", "has_ip", "has_https", "has_port", "is_shortened",
    "suspicious_word_count", "digit_ratio", "url_entropy", "host_entropy",
    "query_length", "has_query", "is_suspicious_tld", "has_punycode",
    "has_double_slash_redirect", "brand_edit_distance", "ends_with_known_ext",
    "longest_word_length",
]

URL_NETWORK_KEYS = ["has_dns_record", "domain_age_days"]

# ---------------------------------------------------------------------------
# HTML/DOM feature schema (order is canonical)
# ---------------------------------------------------------------------------
HTML_NUMERIC_KEYS = [
    "num_links", "external_link_ratio", "empty_link_ratio", "unsafe_anchor_ratio",
    "num_forms", "external_form_ratio", "sfh_suspicious", "mail_submission",
    "num_password_fields", "num_hidden_fields",
    "num_scripts", "external_script_ratio", "external_css_ratio",
    "num_iframes", "invisible_iframe_count", "frameborder_zero_count",
    "num_images", "internal_media_ratio", "external_media_ratio",
    "num_meta_tags", "has_title", "title_length", "text_length",
    "favicon_external", "has_right_click_disabled", "has_status_bar_change",
    "has_popup", "has_popup_with_input", "copyright_mismatch",
    "num_redirects_meta", "char_to_tag_ratio",
]

# ---------------------------------------------------------------------------
# LLM semantic signal schema
# ---------------------------------------------------------------------------
SEMANTIC_SCORE_KEYS = [
    "urgency_language",
    "brand_impersonation",
    "credential_harvesting",
    "generic_greeting",
    "grammar_anomaly_score",
    "mismatched_domain_claim",
]

# ---------------------------------------------------------------------------
# Text (char-level TF-IDF) configuration
# ---------------------------------------------------------------------------
CHAR_NGRAM_RANGE = (2, 4)
CHAR_MAX_FEATURES_URL = 512
CHAR_MAX_FEATURES_TEXT = 512

# ---------------------------------------------------------------------------
# Fusion / training configuration
# ---------------------------------------------------------------------------
MODEL_DIM = 64
ATTENTION_HIDDEN = 64
TRAIN_EPOCHS = 80
TRAIN_BATCH_SIZE = 64
TRAIN_LR = 1e-3
RANDOM_SEED = 42

HTTP_TIMEOUT = 10
HTTP_MAX_BYTES = 2_000_000
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Maximum HTML characters retained per page (guards memory/parse time on large pages)
HTML_MAX_CHARS = 120_000
