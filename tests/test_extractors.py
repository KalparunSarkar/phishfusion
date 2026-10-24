import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from url_feature_extractor import (  # noqa: E402
    extract_url_features,
    registered_domain,
    url_feature_vector,
)
from html_feature_extractor import extract_html_features, html_feature_vector  # noqa: E402
from config import HTML_NUMERIC_KEYS, URL_NUMERIC_KEYS  # noqa: E402


def test_registered_domain_extraction():
    assert registered_domain("secure-login.paypal.com.verify.xyz") == "verify.xyz"
    assert registered_domain("www.google.com") == "google.com"
    assert registered_domain("a.b.example.co.uk") == "example.co.uk"


def test_url_vector_canonical_length_and_order():
    feats = extract_url_features("http://secure-login.paypal.com.verify.xyz/webscr?cmd=login")
    vec = url_feature_vector(feats)
    assert len(vec) == len(URL_NUMERIC_KEYS)
    assert set(URL_NUMERIC_KEYS).issubset(feats.keys())


def test_brand_edit_distance_targets_registered_domain():
    # registered domain is "paypa1.com"-style; distance to "paypal" should be small
    feats = extract_url_features("http://login.paypa1.com/")
    assert feats["brand_edit_distance"] == 1


def test_ip_address_detection_and_rejection():
    assert extract_url_features("http://192.168.0.1/login")["has_ip"] == 1
    assert extract_url_features("http://999.1.1.1/login")["has_ip"] == 0


def test_suspicious_tld_and_punycode():
    assert extract_url_features("http://a.b.xyz/")["is_suspicious_tld"] == 1
    assert extract_url_features("http://xn--paypa1-9db.com/")["has_punycode"] == 1


def test_html_vector_length_and_invisible_iframe_fix():
    html = """
    <html><head><title>t</title></head><body>
    <iframe frameborder="0" src="http://evil.com/a"></iframe>
    <iframe style="display:none" src="http://evil.com/b"></iframe>
    <form action=""><input type="password"></form>
    </body></html>
    """
    feats, text = extract_html_features(html, "http://phish.xyz")
    assert len(html_feature_vector(feats)) == len(HTML_NUMERIC_KEYS)
    # frameborder=0 must NOT count as invisible; only the display:none one should
    assert feats["invisible_iframe_count"] == 1
    assert feats["frameborder_zero_count"] == 1
    assert feats["num_password_fields"] == 1
    assert feats["sfh_suspicious"] == 1


def test_external_link_uses_registered_domain():
    html = '<a href="https://www.paypal.com/x">a</a><a href="http://evil.xyz/b">b</a>'
    feats, _ = extract_html_features(f"<html><body>{html}</body></html>", "https://paypal.com/")
    # www.paypal.com is same registered domain -> only evil.xyz is external
    assert feats["external_link_ratio"] == 0.5
