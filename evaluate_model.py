from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_DIR = "./models/hebrew-phishing-model"

DATA_FILES = [
    "hebrew_dataset.csv",
    "hebrew_dataset_large_normalized.csv",
]

dataset = load_dataset("csv", data_files={"data": DATA_FILES})
dataset = dataset["data"].train_test_split(test_size=0.2, seed=42)
test_ds = dataset["test"]

label2id = {"benign": 0, "phishing": 1}
id2label = {v: k for k, v in label2id.items()}

def encode_labels(batch):
    batch["labels"] = [label2id[l] for l in batch["label"]]
    return batch

test_ds = test_ds.map(encode_labels, batched=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

test_enc = test_ds.map(preprocess, batched=True)
test_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=test_enc,
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate()

print("\n=== Evaluation Metrics ===")
print(f"Accuracy:  {metrics['eval_accuracy']:.4f} ({metrics['eval_accuracy']*100:.2f}%)")
print(f"Precision: {metrics['eval_precision']:.4f} ({metrics['eval_precision']*100:.2f}%)")
print(f"Recall:    {metrics['eval_recall']:.4f} ({metrics['eval_recall']*100:.2f}%)")
print(f"F1-score:  {metrics['eval_f1']:.4f} ({metrics['eval_f1']*100:.2f}%)")