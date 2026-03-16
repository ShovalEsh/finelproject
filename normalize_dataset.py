import pandas as pd

BIG_IN = "hebrew_dataset_large.csv"
SMALL_IN = "hebrew_dataset.csv"
BIG_OUT = "hebrew_dataset_large_normalized.csv"

small = pd.read_csv(SMALL_IN, encoding="utf-8")
target_cols = list(small.columns)

big = pd.read_csv(BIG_IN, encoding="utf-8")

if "text" not in big.columns:
    raise ValueError(f"Big dataset must contain 'text' column. Found: {big.columns}")

def to_label(x):
    if pd.isna(x):
        return "benign"
    s = str(x).strip().lower()

    if s.replace(".", "", 1).isdigit():
        return "phishing" if int(float(s)) == 1 else "benign"

    return "phishing" if s in {"phishing", "spam", "scam", "fraud", "malicious"} else "benign"

if "label" not in big.columns:
    raise ValueError("Big dataset must contain 'label' column")

big["label"] = big["label"].apply(to_label)

if "risk_type" in big.columns:
    pass
elif "top_risk" in big.columns:
    big["risk_type"] = big["top_risk"].fillna("unknown").astype(str)
else:
    big["risk_type"] = "unknown"

out = big.reindex(columns=target_cols)

for col in out.columns:
    out[col] = out[col].fillna("")

out.to_csv(BIG_OUT, index=False, encoding="utf-8")
print("✅ Saved:", BIG_OUT)
print("✅ Columns:", list(out.columns))
print("✅ Rows:", len(out))
