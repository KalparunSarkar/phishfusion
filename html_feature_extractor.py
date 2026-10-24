"""HTML/DOM structural feature extractor.

Corrects two defects present in the earlier version:

* ``frameborder="0"`` is no longer counted as an *invisible* iframe - it only
  removes the border.  A dedicated ``frameborder_zero_count`` feature is
  reported instead, and invisibility is judged from style/attributes;
* external/internal comparisons use *registered* domains, so a page linking to
  ``www.paypal.com`` from ``paypal.com`` is not falsely flagged as external;
* the copyright/brand mismatch check compares against the registered domain.
"""

from urllib.parse import urlparse

from bs4 import BeautifulSoup

from config import HTML_NUMERIC_KEYS
from url_feature_extractor import registered_domain

_EMPTY_HREF = {"", "#", "javascript:void(0)", "javascript:;", "javascript:void(0);"}


def _host_of(url):
    try:
        return urlparse(url if "://" in url else "http://" + url).netloc.split(":")[0].lower()
    except Exception:
        return ""


def _is_external(target, page_host):
    """True when target resolves to a different registered domain."""
    th = _host_of(target)
    if not th:
        return False
    return registered_domain(th) != registered_domain(page_host)


def _is_invisible_iframe(tag):
    style = (tag.get("style") or "").replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    if tag.has_attr("hidden"):
        return True
    w = (tag.get("width") or "").strip()
    h = (tag.get("height") or "").strip()
    if w in ("0", "0px") or h in ("0", "0px"):
        return True
    return False


def extract_html_features(html, page_url=""):
    """Return ``(features, visible_text)`` for a rendered HTML document."""
    soup = BeautifulSoup(html or "", "html.parser")
    page_host = _host_of(page_url)

    anchors = soup.find_all("a", href=True)
    forms = soup.find_all("form")
    scripts = soup.find_all("script")
    iframes = soup.find_all("iframe")
    inputs = soup.find_all("input")
    metas = soup.find_all("meta")
    images = soup.find_all("img")
    stylesheets = soup.find_all("link", rel=lambda v: v and "stylesheet" in _as_list(v))

    total_links = len(anchors)
    external_links = 0
    empty_links = 0
    for a in anchors:
        href = (a.get("href") or "").strip()
        if href in _EMPTY_HREF:
            empty_links += 1
            continue
        if _is_external(href, page_host):
            external_links += 1

    unsafe_anchors = sum(
        1 for a in anchors
        if (a.get("href") or "").strip().lower().startswith(("#", "javascript", "mailto"))
    )

    password_fields = sum(1 for i in inputs if (i.get("type") or "").lower() == "password")
    hidden_fields = sum(1 for i in inputs if (i.get("type") or "").lower() == "hidden")

    external_forms = 0
    for f in forms:
        action = (f.get("action") or "").strip()
        if _is_external(action, page_host):
            external_forms += 1

    sfh_suspicious = sum(
        1 for f in forms if (f.get("action") or "").strip().lower() in ("", "about:blank")
    )
    html_lower = (html or "").lower()
    mail_submission = sum(
        1 for f in forms
        if (f.get("action") or "").strip().lower().startswith("mailto:")
    ) + (1 if "mail(" in html_lower else 0)

    invisible_iframes = sum(1 for f in iframes if _is_invisible_iframe(f))
    frameborder_zero = sum(1 for f in iframes if str(f.get("frameborder", "")).strip() == "0")

    internal_media = 0
    external_media = 0
    for tag in soup.find_all(["img", "audio", "video", "source"]):
        src = tag.get("src", "")
        if not src:
            continue
        if _is_external(src, page_host):
            external_media += 1
        else:
            internal_media += 1
    total_media = internal_media + external_media

    external_scripts = sum(
        1 for s in scripts if s.get("src") and _is_external(s["src"], page_host)
    )
    external_css = sum(
        1 for l in stylesheets if l.get("href") and _is_external(l["href"], page_host)
    )

    text_content = soup.get_text(separator=" ", strip=True)
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    features = {
        "num_links": total_links,
        "external_link_ratio": external_links / total_links if total_links else 0.0,
        "empty_link_ratio": empty_links / total_links if total_links else 0.0,
        "unsafe_anchor_ratio": unsafe_anchors / total_links if total_links else 0.0,
        "num_forms": len(forms),
        "external_form_ratio": external_forms / len(forms) if forms else 0.0,
        "sfh_suspicious": int(sfh_suspicious > 0),
        "mail_submission": int(mail_submission > 0),
        "num_password_fields": password_fields,
        "num_hidden_fields": hidden_fields,
        "num_scripts": len(scripts),
        "external_script_ratio": external_scripts / len(scripts) if scripts else 0.0,
        "external_css_ratio": external_css / len(stylesheets) if stylesheets else 0.0,
        "num_iframes": len(iframes),
        "invisible_iframe_count": invisible_iframes,
        "frameborder_zero_count": frameborder_zero,
        "num_images": len(images),
        "internal_media_ratio": internal_media / total_media if total_media else 0.0,
        "external_media_ratio": external_media / total_media if total_media else 0.0,
        "num_meta_tags": len(metas),
        "has_title": int(bool(title)),
        "title_length": len(title),
        "text_length": len(text_content),
        "favicon_external": int(_has_external_favicon(soup, page_host)),
        "has_right_click_disabled": int(
            "event.button==2" in html_lower.replace(" ", "")
            or "oncontextmenu" in html_lower
        ),
        "has_status_bar_change": int(
            "onmouseover" in html_lower and "window.status" in html_lower
        ),
        "has_popup": int("window.open(" in html_lower),
        "has_popup_with_input": int("window.open(" in html_lower and "<input" in html_lower),
        "copyright_mismatch": int(_copyright_mismatch(text_content, page_host)),
        "num_redirects_meta": len(
            soup.find_all("meta", attrs={"http-equiv": lambda v: v and v.lower() == "refresh"})
        ),
        "char_to_tag_ratio": len(text_content) / (len(html) or 1),
    }
    return features, text_content


def html_feature_vector(features):
    return [float(features.get(k, 0.0)) for k in HTML_NUMERIC_KEYS]


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def _has_external_favicon(soup, page_host):
    icon = soup.find("link", rel=lambda v: v and "icon" in " ".join(_as_list(v)))
    if not icon or not icon.get("href"):
        return False
    return _is_external(icon["href"], page_host)


def _copyright_mismatch(text_content, page_host):
    if "\u00a9" not in text_content and "(c)" not in text_content.lower():
        return False
    if not page_host:
        return False
    brand = registered_domain(page_host).split(".")[0]
    return brand.lower() not in text_content.lower()


if __name__ == "__main__":
    sample_html = """
    <html><head><title>Verify Your Account</title></head>
    <body>
    <form action=""><input type="password"><input type="hidden" name="x"></form>
    <a href="http://paypal.com">PayPal</a><a href="#">Empty</a>
    <iframe frameborder="0" src="http://evil.com/frame"></iframe>
    <iframe style="display:none" src="http://evil.com/x"></iframe>
    <img src="http://cdn-evil.com/logo.png">
    <script>window.open('popup.html'); document.oncontextmenu=function(){event.button==2}</script>
    <p>&copy; 2024 PayPal</p>
    </body></html>
    """
    feats, text = extract_html_features(sample_html, "http://secure-login.paypal.com.verify.xyz")
    for k, v in feats.items():
        print(f"  {k:28} {v}")
    print("vector len:", len(html_feature_vector(feats)), "| text:", text[:60])
