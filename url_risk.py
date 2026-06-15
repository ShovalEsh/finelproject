from __future__ import annotations
import ipaddress
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import tldextract
from confusable_homoglyphs import confusables

SUSPICIOUS_BRANDS = [
    "facebook",
    "instagram",
    "whatsapp",
    "telegram",
    "google",
    "gmail",
    "paypal",
    "apple",
    "icloud",
    "microsoft",
    "outlook",
    "amazon",
    "netflix",
    "bankhapoalim",
    "hapoalim",
    "leumi",
    "discount",
    "isracard",
    "max",
    "visa",
    "mastercard",
    "bit",
    "pepper",
]

LEET_TRANSLATIONS = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
}

URL_REGEX = re.compile(
    r"""
    (?:
        https?://[^\s<>"'()\[\]]+ |
        www\.[^\s<>"'()\[\]]+ |
        (?<!@)\b[a-zA-Z0-9\u00A1-\uFFFF][a-zA-Z0-9\u00A1-\uFFFF.-]{0,252}
        \.[a-zA-Z\u00A1-\uFFFF]{2,24}(?:/[^\s<>"'()\[\]]*)?
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.UNICODE,
)

TRAILING_PUNCTUATION = '.,;:!?)]}"\''
SUSPICIOUS_TLDS = {
    "top", "xyz", "click", "shop", "live", "support", "country", "stream", "gq"
}

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip().lower()

def deobfuscate_token(token: str) -> str:
    token = normalize_text(token)
    return "".join(LEET_TRANSLATIONS.get(ch, ch) for ch in token)

def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    matches = URL_REGEX.findall(text)
    cleaned: List[str] = []
    for match in matches:
        candidate = match.strip().rstrip(TRAILING_PUNCTUATION)
        if not candidate:
            continue
        cleaned.append(candidate)
    return list(dict.fromkeys(cleaned))

def ensure_scheme(candidate: str) -> str:
    candidate = candidate.strip()
    if not candidate.startswith(("http://", "https://")):
        return "http://" + candidate
    return candidate

def get_host(candidate: str) -> str:
    try:
        parsed = urlparse(ensure_scheme(candidate))
        return (parsed.hostname or "").strip().lower().rstrip(".")
    except Exception:
        return ""

def is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False

def safe_tld_extract(host: str):
    return tldextract.extract(host)

def has_non_ascii(host: str) -> bool:
    return any(ord(ch) > 127 for ch in host)

def has_confusable_chars(host: str) -> bool:
    if not host:
        return False
    try:
        if confusables.is_mixed_script(host):
            return True
        result = confusables.is_confusable(host, preferred_aliases=["latin"])
        return bool(result)
    except Exception:
        return False

def count_subdomains(host: str) -> int:
    ext = safe_tld_extract(host)
    if not ext.subdomain:
        return 0
    return len([p for p in ext.subdomain.split(".") if p])

def brand_spoof_check(host: str) -> Tuple[bool, Optional[str], Optional[str]]:
    ext = safe_tld_extract(host)
    domain = normalize_text(ext.domain)
    subdomain = normalize_text(ext.subdomain)
    full_core = ".".join(part for part in [subdomain, domain] if part)
    if not domain:
        return False, None, None
    domain_deobf = deobfuscate_token(domain)
    full_core_deobf = deobfuscate_token(full_core)

    for brand in SUSPICIOUS_BRANDS:
        if domain == brand:
            continue
        # faceb00k -> facebook
        if domain_deobf == brand:
            return True, brand, "leet_brand_spoof"
        # facebook-login-secure
        if brand in domain_deobf and domain_deobf != brand:
            return True, brand, "brand_embedded_in_domain"
        # facebook.verify-login.badsite.com
        if brand in full_core_deobf and domain != brand:
            return True, brand, "brand_in_subdomain_or_chain"
    return False, None, None

def structural_flags(host: str) -> List[str]:
    flags: List[str] = []
    ext = safe_tld_extract(host)
    if not host:
        return flags
    if host.startswith("xn--") or ".xn--" in host:
        flags.append("punycode_domain")
    if is_ip_host(host):
        flags.append("ip_address_link")
    if host.count("-") >= 2:
        flags.append("many_hyphens")
    if count_subdomains(host) >= 3:
        flags.append("many_subdomains")
    if ext.suffix and ext.suffix.lower() in SUSPICIOUS_TLDS:
        flags.append("suspicious_tld")
    if len(ext.domain or "") >= 22:
        flags.append("very_long_domain")
    return flags

def score_from_flags(flags: List[str]) -> int:
    weights = {
        "confusable_characters": 35,
        "non_ascii_domain": 15,
        "punycode_domain": 25,
        "ip_address_link": 25,
        "many_hyphens": 10,
        "many_subdomains": 10,
        "suspicious_tld": 10,
        "very_long_domain": 8,
        "leet_brand_spoof": 40,
        "brand_embedded_in_domain": 30,
        "brand_in_subdomain_or_chain": 30,
    }
    score = 0
    for flag in flags:
        score += weights.get(flag, 0)
    return min(score, 100)

def flags_to_reasons(flags: List[str], brand: Optional[str] = None) -> List[str]:
    reasons: List[str] = []
    for flag in flags:
        if flag == "confusable_characters":
            reasons.append("Domain contains look-alike characters.")
        elif flag == "non_ascii_domain":
            reasons.append("Domain contains non-ASCII characters.")
        elif flag == "punycode_domain":
            reasons.append("Domain uses punycode encoding.")
        elif flag == "ip_address_link":
            reasons.append("Link uses an IP address instead of a normal domain.")
        elif flag == "many_hyphens":
            reasons.append("Domain contains many hyphens.")
        elif flag == "many_subdomains":
            reasons.append("Domain contains many subdomains.")
        elif flag == "suspicious_tld":
            reasons.append("Domain uses a high-risk top-level domain.")
        elif flag == "very_long_domain":
            reasons.append("Domain name is unusually long.")
        elif flag == "leet_brand_spoof":
            reasons.append(
                f"Domain looks like a spoof of '{brand}' using replaced characters."
                if brand else
                "Domain looks like a spoofed brand using replaced characters."
            )
        elif flag == "brand_embedded_in_domain":
            reasons.append(
                f"Brand name '{brand}' appears embedded in a suspicious domain."
                if brand else
                "A known brand name appears embedded in a suspicious domain."
            )
        elif flag == "brand_in_subdomain_or_chain":
            reasons.append(
                f"Brand name '{brand}' appears in subdomain/path-like structure."
                if brand else
                "A known brand name appears in a misleading subdomain structure."
            )
    return reasons

def analyze_single_url(candidate: str) -> Dict[str, Any]:
    host = get_host(candidate)
    flags: List[str] = []
    brand: Optional[str] = None
    if not host:
        return {
            "candidate": candidate,
            "host": "",
            "score": 0,
            "flags": [],
            "reasons": [],
            "is_suspicious": False,
        }

    if has_non_ascii(host):
        flags.append("non_ascii_domain")

    if has_confusable_chars(host):
        flags.append("confusable_characters")

    spoofed, brand, brand_reason = brand_spoof_check(host)
    if spoofed and brand_reason:
        flags.append(brand_reason)

    flags.extend(structural_flags(host))

    flags = list(dict.fromkeys(flags))

    score = score_from_flags(flags)
    reasons = flags_to_reasons(flags, brand=brand)

    return {
        "candidate": candidate,
        "host": host,
        "score": score,
        "flags": flags,
        "brand": brand,
        "reasons": reasons,
        "is_suspicious": score >= 25,
    }

def analyze_urls_in_text(text: str) -> Dict[str, Any]:
    urls = extract_urls(text)
    analyzed = [analyze_single_url(u) for u in urls]
    suspicious = [item for item in analyzed if item["is_suspicious"]]
    max_score = max((item["score"] for item in analyzed), default=0)
    return {
        "found": bool(urls),
        "urls": urls,
        "analyzed": analyzed,
        "suspicious": suspicious,
        "max_url_risk_score": max_score,
    }