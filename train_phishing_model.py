"""
training hebrew_dataset.csv with DeBERTa
Run:
  python train_phishing_model.py
"""
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate
from sklearn.metrics import f1_score

MODEL_NAME = "microsoft/deberta-v3-xsmall"
DATA_PATH = "hebrew_dataset.csv"
OUTPUT_DIR = "./models/hebrew-phishing-model"

dataset = load_dataset("csv", data_files={"data": DATA_PATH})
dataset = dataset["data"].train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

label2id = {"benign": 0, "phishing": 1}
id2label = {v: k for k, v in label2id.items()}

def encode_labels(batch):
    batch["labels"] = [label2id[l] for l in batch["label"]]
    return batch

train_ds = train_ds.map(encode_labels, batched=True)
test_ds = test_ds.map(encode_labels, batched=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_enc = train_ds.map(preprocess, batched=True)
test_enc = test_ds.map(preprocess, batched=True)

train_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=test_enc,
    compute_metrics=compute_metrics,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=20,
    seed=42,

)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training finished, model saved to", OUTPUT_DIR)
