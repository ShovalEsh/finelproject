from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def main():
    model_name = "avichr/heBERT_sentiment_analysis"

    print("Loading HeBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        return_all_scores=True
    )

    texts = [
        "קיבלת הודעה מהבנק: אנא עדכן את פרטי חשבון הבנק שלך בקישור הבא.",
        "זכית במיליון ש\"ח! לחץ על הקישור כדי לקבל את הפרס.",
        "היי סבתא, רק מזכירה לך לקחת תרופות. אוהבת אותך ❤️",
    ]

    label_map = {
        "LABEL_0": "שלילי",
        "LABEL_1": "נייטרלי",
        "LABEL_2": "חיובי",
    }

    for text in texts:
        print("\nטקסט:")
        print(text)

        results = pipe(text)[0] 
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        best = results[0]
        nice_label = label_map.get(best["label"], best["label"])

        print(f"ניבוי מוביל: {nice_label} (score={best['score']:.3f})")
        print("כל הציונים:")
        for r in results:
            print(f"  {label_map.get(r['label'], r['label'])}: {r['score']:.3f}")

if __name__ == "__main__":
    main()
