from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset

# ds = load_dataset("FredZhang7/toxi-text-3M")

# pipe = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def extract_urls(text: str) -> List[str]:
    return [u.rstrip(").,;!?]\"'") for u in URL_RE.findall(text or "")]

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
        0.8,
    ),
    (
        ["חשבון", "סיסמה", "התחברות", "אימות", "קוד", "בנק"],
        "Asks about account or verification details.",
        0.85,
    ),
    (
        ["קישור", "לינק", "לחץ", "היכנס", "כניסה", "לחיצה"],
        "Encourages clicking a link.",
        0.75,
    ),
    (
        ["דחוף", "מייד", "מיד", "בהקדם"],
        "Uses urgency pressure.",
        0.65,
    ),
]

HEBREW_TOKEN_RE = re.compile(r"[\u0590-\u05FF]+")
HEBREW_PREFIXES = ("ו", "ב", "ל", "מ", "כ", "ה")

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

@dataclass
class RiskResult:
    risk_score: float
    top_risk: str
    risks: Dict[str, float]
    reasons: List[str]
    consequences: List[str]
    urls: List[str]

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

class SklearnSpamModel:
    """
    Loads a trained TF-IDF + LogisticRegression model (English SMS spam).
    """
    def __init__(self, model_path: str = "spam_model.joblib", vectorizer_path: str = "tfidf.joblib"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self._model = None
        self._vec = None

    def _load(self):
        if self._model is not None and self._vec is not None:
            return
        import joblib
        self._model = joblib.load(self.model_path)
        self._vec = joblib.load(self.vectorizer_path)

    def predict_spam_probability(self, text: str) -> float:
        self._load()
        X = self._vec.transform([text])
        proba = self._model.predict_proba(X)[0]
        classes = list(self._model.classes_)
        if "spam" in classes:
            return float(proba[classes.index("spam")])
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

class OpenAIRiskModel:
    """
    Calls OpenAI to classify risk with structured JSON output.
    """
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def predict(self, text: str) -> Dict[str, Any]:
        from openai import OpenAI
        client = OpenAI()
        schema = {
            "type": "object",
            "properties": {
                "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
                "top_risk": {"type": "string"},
                "risks": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "reasons": {"type": "array", "items": {"type": "string"}},
                "consequences": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["risk_score", "top_risk", "risks", "reasons", "consequences"],
            "additionalProperties": False
        }

        prompt = (
            "You are a cybersecurity assistant for older adults. "
            "Given a message they received (SMS/WhatsApp/email), assess scam risk.\n"
            "Return JSON only.\n\n"
            f"Message:\n{text}"
        )

        resp = client.responses.create(
            model=self.model,
            input=prompt,
            response_format={"type": "json_schema", "json_schema": {"name": "risk_assessment", "schema": schema}},
        )
        return resp.output_parsed

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
        sklearn_spam: Optional[SklearnSpamModel] = None,
        openai_model: Optional[OpenAIRiskModel] = None,
        finetuned_phishing: Optional[FineTunedPhishingModel] = None,
    ):
        self.zero_shot = zero_shot
        self.sklearn_spam = sklearn_spam
        self.openai_model = openai_model
        self.finetuned_phishing = finetuned_phishing

    def analyze(self, text: str) -> RiskResult:
        text = (text or "").strip()
        lower_text = text.lower()
        is_hebrew = looks_like_hebrew(text)
        urls = extract_urls(text)
        hebrew_signals = find_hebrew_scam_signals(text) if is_hebrew else []
        hebrew_signal_reasons = [reason for reason, _ in hebrew_signals]
        hebrew_signal_weights = [weight for _, weight in hebrew_signals]
        signal_weights = list(hebrew_signal_weights)
        if urls:
            signal_weights.append(0.8)
        hebrew_signal_score = combine_signal_weights(signal_weights)

        if self.openai_model is not None:
            out = self.openai_model.predict(text)
            return RiskResult(
                risk_score=float(out["risk_score"]),
                top_risk=str(out["top_risk"]),
                risks={k: float(v) for k, v in dict(out["risks"]).items()},
                reasons=list(out["reasons"]),
                consequences=list(out["consequences"]),
                urls=urls,
            )

        risks: Dict[str, float] = {}
        use_zero_shot = self.zero_shot is not None and not (
            is_hebrew and self.finetuned_phishing is not None
        )
        if use_zero_shot:
            risks = self.zero_shot.predict(text)

        spam_p = None
        if self.sklearn_spam is not None and looks_like_english(text):
            try:
                spam_p = self.sklearn_spam.predict_spam_probability(text)
            except Exception:
                spam_p = None

        raw_phishing_p = None
        if self.finetuned_phishing is not None:
            try:
                raw_phishing_p = self.finetuned_phishing.predict_proba(text)
            except Exception:
                raw_phishing_p = None

        phishing_p = raw_phishing_p
        if is_hebrew and hebrew_signal_score:
            if raw_phishing_p is None:
                phishing_p = hebrew_signal_score
            else:
                blended = (0.2 * raw_phishing_p) + (0.8 * hebrew_signal_score)
                phishing_p = max(raw_phishing_p, blended)

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
        has_hard_signal = (
            bool(urls)
            or ("otp" in lower_text or "קוד" in text or "אימות" in text)
            or ("bank" in lower_text or "בנק" in text)
            or ("urgent" in lower_text or "דחוף" in text or "מיד" in text)
            or bool(hebrew_signal_reasons)
        )

        force_benign = is_clearly_benign or ((not has_hard_signal) and risk_score < 0.75)

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
            # risk_score = min(1.0, risk_score + 0.10)
            reasons.append("Contains a link (common in phishing/malware).")
        if "otp" in lower_text or "קוד" in text or "אימות" in text:
            reasons.append("Asks for a verification code/OTP (often account takeover).")
        if "bank" in lower_text or "בנק" in text:
            reasons.append("Mentions a bank/account (common in phishing).")
        if "urgent" in lower_text or "דחוף" in text or "מיד" in text:
            reasons.append("Uses urgency pressure.")
        if hebrew_signal_reasons:
            for reason in hebrew_signal_reasons:
                if reason not in reasons:
                    reasons.append(reason)

        consequences = CONSEQUENCE_MAP.get(top_risk, ["Possible scam impact: money loss, account takeover, or malware."])

        # risks_sorted = dict(sorted(risks.items(), key=lambda kv: kv[1], reverse=True)) if risks else {}
        # return RiskResult(
        #     risk_score=float(risk_score),
        #     top_risk=top_risk,
        #     risks=risks_sorted,
        #     reasons=reasons[:6],
        #     consequences=consequences[:6],
        #     urls=urls,
        # )

        risks_sorted = (
            {k: round(v, 2) for k, v in sorted(risks.items(), key=lambda kv: kv[1], reverse=True)}
            if risks else {}
        )

        return RiskResult(
            risk_score=round(float(risk_score), 2),
            top_risk=top_risk,
            risks=risks_sorted,
            reasons=reasons[:6],
            consequences=consequences[:6],
            urls=urls,
        )

