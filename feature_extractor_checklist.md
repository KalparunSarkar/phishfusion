# Feature Extractor Checklist

## URL Feature Extractor

- URL, host, and path length
- Count of dots, hyphens, underscores, slashes
- Digit count and digit ratio
- Count of `@`, `?`, `=`, `%`, `&`, `$`, `!`, `#`
- Number of subdomains
- IP address used instead of domain name
- HTTPS presence
- Non-standard port usage
- URL shortening service detection
- Suspicious keyword count (login, verify, secure, bank, etc.)
- URL entropy and host entropy (randomness)
- Query string presence and length
- Top-level domain (TLD) extraction
- Suspicious TLD detection (.xyz, .tk, .gq, etc.)
- Punycode / IDN homograph detection
- Double-slash redirect trick in path
- Brand typosquatting via edit distance to popular brand names
- Suspicious file extension at end of path (.exe, .php)
- Longest token length in URL
- Character-level n-grams (for TF-IDF/embedding input)
- *(optional, network-based)* DNS record existence
- *(optional, network-based)* Domain age via WHOIS

## HTML/DOM Feature Extractor

- Total number of hyperlinks
- External link ratio
- Empty/self-referencing link ratio
- Unsafe anchor ratio (`#`, `javascript:`, `mailto:`)
- Number of forms
- External form-action ratio
- Suspicious Server Form Handler (empty/`about:blank`)
- Mail-based data submission (`mailto:`, PHP `mail()`)
- Password field count
- Hidden field count
- Script tag count and external script ratio
- External CSS ratio
- Iframe count and invisible iframe count
- Image count
- Internal vs. external media ratio
- Meta tag count
- Page title presence and length
- Visible text length
- External favicon detection
- Right-click disabling detection
- Status bar spoofing (`onMouseOver` + `window.status`)
- Popup window detection
- Popup with input fields (credential harvesting)
- Copyright/brand mismatch in page text
- Meta-refresh redirect count
- Text-to-HTML tag ratio
- Extracted plaintext (passed downstream for LLM semantic analysis)
