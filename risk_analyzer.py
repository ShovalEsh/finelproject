from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from url_risk import analyze_urls_in_text
import unicodedata

def generate_recommendations(risk_score, reasons, urls):
    recommendations = []
    reasons_text = " ".join(reasons).lower()
    if risk_score < 0.3:
        recommendations.append("SAFE_NO_ACTION")
        recommendations.append("SAFE_STAY_ALERT")
        return recommendations
    if urls or "link" in reasons_text or "url" in reasons_text:
        recommendations.append("AVOID_LINKS")
    if "urgency" in reasons_text or "דחוף" in reasons_text or "urgent" in reasons_text:
        recommendations.append("DO_NOT_RUSH")
    if (
        "verification" in reasons_text
        or "otp" in reasons_text
        or "code" in reasons_text
        or "קוד" in reasons_text
        or "אימות" in reasons_text
    ):
        recommendations.append("DO_NOT_SHARE_INFO")
    if (
        "debt" in reasons_text
        or "payment" in reasons_text
        or "fine" in reasons_text
        or "settlement" in reasons_text
        or "חוב" in reasons_text
        or "תשלום" in reasons_text
    ):
        recommendations.append("VERIFY_PAYMENT_OFFICIAL_SOURCE")
    if (
        "known organization" in reasons_text
        or "trusted entity" in reasons_text
        or "impersonation" in reasons_text
    ):
        recommendations.append("CONTACT_OFFICIAL_CHANNEL")
    if risk_score >= 0.7:
        recommendations.append("BLOCK_AND_REPORT")
    if not recommendations:
        recommendations.append("BE_CAREFUL")
    return list(dict.fromkeys(recommendations))

def normalize_message_text(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def looks_like_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in (text or ""))

def looks_like_english(text: str) -> bool:
    letters = sum(("a" <= ch.lower() <= "z") for ch in (text or ""))
    return letters >= max(10, int(0.15 * max(1, len(text))))

HEBREW_SCAM_SIGNALS: List[Tuple[List[str], str, float]] = [
    (
        ["זכית", "זכייה", "פרס", "הגרלה", "מיליון", "דולר", "מתנה", "בונוס"],
        "Mentions a prize/lottery (common scam).",
        0.75,
    ),
    (
        ["תשלום", "העברה", "הפקדה", "חיוב", "כרטיס", "אשראי", "חשבונית"],
        "Requests money or payment details.",
        0.80,
    ),
    (
        ["חוב", "חוב פתוח", "אגרה", "קנס", "יתרה", "הסדר", "הסדרת חוב", "תשלום חוב"],
        "Mentions debt, fine, or urgent payment settlement.",
        0.82,
    ),
    (
        ["חשבון", "סיסמה", "התחברות", "אימות", "קוד", "בנק"],
        "Asks about account or verification details.",
        0.85,
    ),
    (
        ["קישור", "לינק", "לחץ", "היכנס", "כניסה", "לחיצה", "לפרטים נוספים"],
        "Encourages clicking a link.",
        0.75,
    ),
    (
        ["דחוף", "מייד", "מיד", "בהקדם", "עוד היום", "היום", "התראה", "אחרון"],
        "Uses urgency pressure.",
        0.72,
    ),
    (
        ["כביש 6", "דואר ישראל", "ביטוח לאומי", "רשות המסים", "משטרה", "בנק", "ויזה", "פייפאל"],
        "Mentions a known organization or trusted entity.",
        0.55,
    ),
]

HEBREW_TOKEN_RE = re.compile(r"[\u0590-\u05FF]+")
HEBREW_PREFIXES = ("ו", "ב", "ל", "מ", "כ", "ה")
SHORTENER_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "rb.gy",
    "rebrand.ly",
    "shorturl.at",
}

def hebrew_tokens(text: str) -> List[str]:
    return HEBREW_TOKEN_RE.findall(text or "")

def token_matches_keyword(token: str, keyword: str) -> bool:
    if not token or not keyword:
        return False
    candidates = [token, token[::-1]]
    for candidate in candidates:
        if candidate == keyword or candidate == keyword[::-1]:
            return True
        if len(candidate) > 2 and candidate[0] in HEBREW_PREFIXES:
            stripped = candidate[1:]
            if stripped == keyword or stripped == keyword[::-1]:
                return True
    return False

def tokens_contain_phrase(tokens: List[str], phrase: str) -> bool:
    words = [word for word in phrase.split() if word]
    if not words:
        return False
    if len(words) == 1:
        word = words[0]
        return any(token_matches_keyword(token, word) for token in tokens)
    for i in range(len(tokens) - len(words) + 1):
        if all(token_matches_keyword(tokens[i + j], words[j]) for j in range(len(words))):
            return True
    return False

def find_hebrew_scam_signals(text: str) -> List[Tuple[str, float]]:
    tokens = hebrew_tokens(text)
    if not tokens:
        return []
    signals: Dict[str, float] = {}
    for keywords, reason, weight in HEBREW_SCAM_SIGNALS:
        match_count = 0
        for keyword in keywords:
            if tokens_contain_phrase(tokens, keyword):
                match_count += 1
        if match_count:
            adjusted_weight = min(0.95, weight + 0.05 * (match_count - 1))
            signals[reason] = max(signals.get(reason, 0.0), adjusted_weight)
    has_verify = tokens_contain_phrase(tokens, "אמת")
    if has_verify:
        for detail_word in ["פרט", "פרטים", "פרטי", "פרטיך", "פרטיכם"]:
            if tokens_contain_phrase(tokens, detail_word):
                signals["Asks about account or verification details."] = max(
                    signals.get("Asks about account or verification details.", 0.0),
                    0.85,
                )
                break
    return list(signals.items())

def combine_signal_weights(weights: List[float]) -> float:
    score = 0.0
    for weight in weights:
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        value = max(0.0, min(1.0, value))
        score = 1.0 - (1.0 - score) * (1.0 - value)
    return max(0.0, min(1.0, score))

def contains_shortened_url(text: str, urls: List[str]) -> bool:
    lowered = (text or "").lower()

    for domain in SHORTENER_DOMAINS:
        if domain in lowered:
            return True

    for url in urls:
        if any(domain in url.lower() for domain in SHORTENER_DOMAINS):
            return True

    return False

@dataclass
class RiskResult:
    risk_score: float
    alert_level: str
    top_risk: str
    message_category: str
    risks: Dict[str, float]
    reasons: List[str]
    recommendations: List[str]
    consequences: List[str]
    urls: List[str]
    suspicious_urls: List[Dict[str, Any]]

class ZeroShotScamModel:
    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(self, model_name: str | None = None, device: int = -1):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._pipe = None
        self.labels = [
            "benign",
            "phishing",
            "financial scam",
            "impersonation",
            "malware or unsafe link",
            "lottery or prize scam",
            "account takeover / OTP request",
            "payment request",
        ]

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )

    def predict(self, text: str) -> Dict[str, float]:
        self._load()
        out = self._pipe(
            text,
            candidate_labels=self.labels,
            multi_label=True,
        )
        return {lbl: float(score) for lbl, score in zip(out["labels"], out["scores"])}


CONSEQUENCE_MAP: Dict[str, List[str]] = {
    "benign": [
        "No scam indicators detected in this message.",
        "If anything feels off, avoid sharing codes or clicking links.",
    ],
    "phishing": [
        "Steal login credentials (bank, email, social networks).",
        "Use the account to scam contacts or steal money.",
    ],
    "financial scam": [
        "Direct money loss via transfer, credit card, or fake invoice.",
        "Identity theft using personal details you share.",
    ],
    "impersonation": [
        "Trick you into sending money/OTP to a fake 'family/bank/support' person.",
        "Harvest private info for future scams.",
    ],
    "malware or unsafe link": [
        "Install malware or remote-control apps on your phone/computer.",
        "Steal passwords, photos, and banking access.",
    ],
    "lottery or prize scam": [
        "Pay 'fees' or provide card details to claim a fake prize.",
        "Ongoing harassment for more payments.",
    ],
    "account takeover / OTP request": [
        "Take over WhatsApp/Telegram/email using your verification code.",
        "Scam your contacts while pretending to be you.",
    ],
    "payment request": [
        "Charge your card or push you to transfer funds.",
        "Lock you into subscriptions or recurring charges.",
    ],
}

class FineTunedPhishingModel:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        return float(probs[1])

class RiskAnalyzer:
    def __init__(
        self,
        zero_shot: Optional[ZeroShotScamModel] = None,
        finetuned_phishing: Optional[FineTunedPhishingModel] = None,
    ):
        self.zero_shot = zero_shot
        self.finetuned_phishing = finetuned_phishing

    def analyze(self, text: str) -> RiskResult:
        text = normalize_message_text(text)
        lower_text = text.lower()
        is_hebrew = looks_like_hebrew(text)
        url_analysis = analyze_urls_in_text(text)
        urls = url_analysis["urls"]
        suspicious_urls = url_analysis["suspicious"]
        url_risk_score = url_analysis["max_url_risk_score"] / 100.0
        has_shortened_url = contains_shortened_url(text, urls)
        hebrew_signals = find_hebrew_scam_signals(text) if is_hebrew else []
        hebrew_signal_reasons = [reason for reason, _ in hebrew_signals]
        hebrew_signal_weights = [weight for _, weight in hebrew_signals]
        signal_weights = list(hebrew_signal_weights)
        if urls:
            signal_weights.append(0.35)
        if has_shortened_url:
            signal_weights.append(0.55)
        if url_risk_score > 0:
            signal_weights.append(url_risk_score)
        hebrew_signal_score = combine_signal_weights(signal_weights)
        trusted_entity_terms = [
            "כביש 6", "דואר ישראל", "ביטוח לאומי", "רשות המסים",
            "משטרה", "בנק", "ויזה", "פייפאל"
        ]

        has_trusted_entity = any(term in text for term in trusted_entity_terms)
        has_payment_or_debt = any(term in text for term in ["חוב", "תשלום", "אגרה", "קנס", "הסדר", "חיוב"])
        has_urgency = any(term in text for term in ["דחוף", "מייד", "מיד", "בהקדם", "עוד היום", "היום", "אחרון"])

        rules_bonus = 0.0

        if has_shortened_url:
            rules_bonus = max(rules_bonus, 0.55)

        if has_payment_or_debt and has_urgency:
            rules_bonus = max(rules_bonus, 0.60)

        if has_trusted_entity and (has_shortened_url or suspicious_urls):
            rules_bonus = max(rules_bonus, 0.75)

        if has_trusted_entity and has_payment_or_debt and (urls or has_shortened_url):
            rules_bonus = max(rules_bonus, 0.80)

        risks: Dict[str, float] = {}
        use_zero_shot = self.zero_shot is not None and not (
            is_hebrew and self.finetuned_phishing is not None
        )
        if use_zero_shot:
            risks = self.zero_shot.predict(text)

        spam_p = None

        raw_phishing_p = None
        if self.finetuned_phishing is not None:
            try:
                raw_phishing_p = self.finetuned_phishing.predict_proba(text)
            except Exception:
                raw_phishing_p = None

        phishing_p = raw_phishing_p
        if is_hebrew:
            combined_rule_score = max(hebrew_signal_score, rules_bonus)
            if combined_rule_score:
                if raw_phishing_p is None:
                    phishing_p = combined_rule_score
                else:
                    blended = (0.55 * raw_phishing_p) + (0.45 * combined_rule_score)
                    phishing_p = max(raw_phishing_p, blended, combined_rule_score * 0.9)

        if raw_phishing_p is not None:
            risks["phishing_supervised"] = raw_phishing_p
        if phishing_p is not None and is_hebrew and not use_zero_shot:
            benign_score = max(0.0, min(1.0, 1.0 - phishing_p))
            risks["phishing"] = phishing_p
            risks["benign"] = benign_score

        benign = float(risks.get("benign", 0.0)) if risks else 0.0
        scam_scores = {
            k: v for k, v in risks.items() if k not in {"benign", "phishing_supervised"}
        }
        scam_max = max(scam_scores.values(), default=0.0)

        reasons: List[str] = []

        is_clearly_benign = (
            benign >= 0.7
            and scam_max < 0.6
            and (phishing_p is None or phishing_p < 0.5)
            and not urls
            and (spam_p is None or spam_p < 0.5)
        )

        candidates = [scam_max]
        if phishing_p is not None:
            candidates.append(phishing_p)
        if spam_p is not None:
            candidates.append(spam_p)

        risk_score = max(candidates) if candidates else 0.0
        if suspicious_urls:
            risk_score = max(risk_score, min(1.0, 0.35 + url_risk_score * 0.65))
    
        has_hard_signal = (
            bool(suspicious_urls)
            or has_shortened_url
            or ("otp" in lower_text or "קוד" in text or "אימות" in text)
            or ("bank" in lower_text or "בנק" in text)
            or has_urgency
            or has_payment_or_debt
            or (has_trusted_entity and (urls or has_shortened_url))
            or bool(hebrew_signal_reasons)
        )

        force_benign = is_clearly_benign or (
            (not has_hard_signal)
            and risk_score < 0.45
            and (phishing_p is None or phishing_p < 0.5)
        )
        if has_trusted_entity and has_payment_or_debt and (has_shortened_url or suspicious_urls or urls):
            force_benign = False

        top_risk = None
        if is_hebrew and not has_hard_signal and phishing_p is not None:
            if phishing_p < 0.5:
                force_benign = True
            else:
                force_benign = False
                risk_score = phishing_p
                top_risk = "phishing"

        if force_benign:
            risk_score = 0.0
            top_risk = "benign"
            reasons.append("No strong scam indicators detected.")
        elif top_risk is None and phishing_p is not None and phishing_p == risk_score and phishing_p >= 0.5:
            top_risk = "phishing"
        elif top_risk is None:
            top_risk = max(
                scam_scores.keys(),
                key=lambda k: scam_scores[k],
                default="benign",
            )

        if force_benign and risks:
            adjusted = {}
            for key, value in risks.items():
                if key == "benign":
                    adjusted[key] = max(value, 0.8)
                else:
                    adjusted[key] = min(value, 0.49)
            risks = adjusted

        if urls:
            if suspicious_urls:
                reasons.append("Contains a suspicious link or spoofed domain.")
                for item in suspicious_urls[:2]:
                    for reason in item.get("reasons", [])[:2]:
                        if reason not in reasons:
                            reasons.append(reason)
            else:
                reasons.append("Contains a link.")

        if has_shortened_url:
            reasons.append("Contains a shortened link, which is commonly used to hide the real destination.")

        if "otp" in lower_text or "קוד" in text or "אימות" in text:
            reasons.append("Asks for a verification code/OTP (often account takeover).")

        if "bank" in lower_text or "בנק" in text:
            reasons.append("Mentions a bank/account (common in phishing).")

        if has_urgency:
            reasons.append("Uses urgency pressure.")

        if has_payment_or_debt:
            reasons.append("Mentions debt, payment, fine, or settlement.")

        if has_trusted_entity:
            reasons.append("Mentions a known organization or trusted entity.")

        if has_trusted_entity and (has_shortened_url or suspicious_urls):
            reasons.append("Possible impersonation of a trusted entity using a suspicious or shortened link.")

        if has_payment_or_debt and has_urgency:
            reasons.append("Combines payment pressure with urgency.")

        if hebrew_signal_reasons:
            for reason in hebrew_signal_reasons:
                if reason not in reasons:
                    reasons.append(reason)

        consequences = CONSEQUENCE_MAP.get(top_risk, ["Possible scam impact: money loss, account takeover, or malware."])

        display_risks = {
        k: v for k, v in risks.items() if k != "phishing_supervised"}
        
        risks_sorted = (
        {k: round(v, 2) for k, v in sorted(display_risks.items(), key=lambda kv: kv[1], reverse=True)}
        if display_risks else {}
        )

        reasons = list(dict.fromkeys(reasons))
        
        if risk_score >= 0.75:
            alert_level = "red"
        elif risk_score >= 0.4:
            alert_level = "yellow"
        else:
            alert_level = "green"

        ui_suspicious_urls = [
            {
                "candidate": item.get("candidate"),
                "host": item.get("host"),
                "score": item.get("score"),
                "reasons": item.get("reasons", [])[:2],
            }
            for item in suspicious_urls[:3]
        ]

        if risk_score >= 0.75:
            message_category = "dangerous"
        elif risk_score >= 0.4:
            message_category = "suspicious"
        else:
            message_category = "safe"

        recommendations = generate_recommendations(
            risk_score=risk_score,
            reasons=reasons,
            urls=urls
        )

        return RiskResult(
        risk_score=round(float(risk_score), 2),
        alert_level=alert_level,
        top_risk=top_risk,
        message_category=message_category,
        risks=risks_sorted,
        reasons=reasons[:6],
        recommendations=recommendations[:6],
        consequences=consequences[:6],
        urls=urls,
        suspicious_urls=ui_suspicious_urls,
        )
