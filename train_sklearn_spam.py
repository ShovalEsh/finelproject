"""
train_sklearn_spam.py
Replaces the current dataset training code, but *saves* the trained artifacts so the API can reuse them.
Run:
  python train_sklearn_spam.py
Outputs:
  tfidf.joblib
  spam_model.joblib
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("spam.csv", sep=",", header=0, encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})
df = df[["label", "message"]]

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(vectorizer, "tfidf.joblib")
joblib.dump(model, "spam_model.joblib")
print("\nSaved: tfidf.joblib, spam_model.joblib")
