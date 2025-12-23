"""
train_hf_classifier.py
Fine-tune a Hugging Face classifier on hebrew_dataset2.csv + spam.csv.

Defaults:
  - Hebrew CSV: risk_type/text columns (utf-8-sig to handle BOM)
  - Spam CSV: v1 (label), v2 (text), map spam->hebrew label, ham->hebrew label
  - Spam rows are downsampled to match Hebrew size (prevents skew)
  - Optional HF dataset: gravitee-io/textdetox-multilingual-toxicity-dataset
  - Use --no-spam to train only on the Hebrew dataset

Run:
  python train_hf_classifier.py --hebrew-data hebrew_dataset2.csv --spam-data spam.csv
Outputs:
  hf_finetuned_model/
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch


def load_hebrew_data(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing columns in {path}: {text_col}, {label_col}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    return df.dropna()


def load_spam_data(
    path: Path,
    text_col: str,
    label_col: str,
    spam_label: str,
    ham_label: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing columns in {path}: {text_col}, {label_col}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    label_map = {"spam": spam_label, "ham": ham_label}
    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    return df.dropna()


def load_toxicity_data(
    dataset_name: str,
    config_name: str,
    split: str,
    text_col: str,
    label_col: str,
    language: str,
    toxic_label: str,
    clean_label: str,
    skip_clean: bool,
    max_rows: int,
) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing 'datasets' package. Install with: pip install datasets") from exc

    dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
    rows: List[Dict[str, str]] = []
    for item in dataset:
        if language:
            lang = str(item.get("language", "")).strip()
            if lang != language:
                continue
        text = str(item.get(text_col, "")).strip()
        if not text:
            continue
        raw_label = item.get(label_col)
        try:
            is_toxic = int(raw_label) == 1
        except (TypeError, ValueError):
            continue
        if not is_toxic and skip_clean:
            continue
        label = toxic_label if is_toxic else clean_label
        if not label:
            continue
        rows.append({"text": text, "label": label})
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def load_label_map(path: Path) -> Dict[str, str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Label map must be a JSON object of {old_label: new_label}.")
    cleaned: Dict[str, str] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        cleaned[str(key).strip()] = str(value).strip()
    return cleaned


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError("Model did not return logits for loss computation.")
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a HF classifier on Hebrew + spam datasets.")
    parser.add_argument("--hebrew-data", default="hebrew_dataset2.csv")
    parser.add_argument("--hebrew-text-col", default="text")
    parser.add_argument("--hebrew-label-col", default="risk_type")
    parser.add_argument("--spam-data", default="spam.csv")
    parser.add_argument("--no-spam", action="store_true")
    parser.add_argument("--spam-text-col", default="v2")
    parser.add_argument("--spam-label-col", default="v1")
    parser.add_argument("--spam-label", default="\u05d4\u05ea\u05d7\u05d6\u05d5\u05ea")
    parser.add_argument("--ham-label", default="\u05ea\u05e7\u05d9\u05df")
    parser.add_argument("--spam-max-rows", type=int, default=0)
    parser.add_argument("--toxicity-dataset", default="")
    parser.add_argument("--toxicity-config", default="default")
    parser.add_argument("--toxicity-split", default="train")
    parser.add_argument("--toxicity-text-col", default="text")
    parser.add_argument("--toxicity-label-col", default="toxic")
    parser.add_argument("--toxicity-language", default="he")
    parser.add_argument("--toxicity-toxic-label", default="\u05d0\u05d9\u05d5\u05dd")
    parser.add_argument("--toxicity-clean-label", default="")
    parser.add_argument("--toxicity-max-rows", type=int, default=0)
    parser.add_argument("--toxicity-skip-clean", action="store_true")
    parser.add_argument("--label-map", default="")
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--rare-strategy", choices=["drop", "merge", "keep"], default="drop")
    parser.add_argument("--rare-label", default="other")
    parser.add_argument("--model-name", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    parser.add_argument("--output-dir", default="hf_finetuned_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--metric-for-best", default="f1_macro")
    parser.add_argument("--class-weighting", action="store_true")
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hebrew_path = Path(args.hebrew_data)
    spam_path = Path(args.spam_data) if args.spam_data else None
    if not hebrew_path.exists():
        raise SystemExit(f"Missing Hebrew dataset: {hebrew_path}")
    if not args.no_spam and spam_path is not None:
        if not spam_path.exists() or spam_path.is_dir():
            raise SystemExit(f"Missing spam dataset: {spam_path}")

    hebrew_df = load_hebrew_data(hebrew_path, args.hebrew_text_col, args.hebrew_label_col)
    if args.no_spam or not args.spam_data:
        spam_df = pd.DataFrame(columns=["text", "label"])
    else:
        spam_df = load_spam_data(
            spam_path,
            args.spam_text_col,
            args.spam_label_col,
            args.spam_label,
            args.ham_label,
        )
    toxicity_df = pd.DataFrame(columns=["text", "label"])
    if args.toxicity_dataset:
        clean_label = args.toxicity_clean_label or args.ham_label
        if args.toxicity_max_rows <= 0:
            toxicity_max_rows = len(hebrew_df)
        else:
            toxicity_max_rows = args.toxicity_max_rows
        toxicity_df = load_toxicity_data(
            args.toxicity_dataset,
            args.toxicity_config,
            args.toxicity_split,
            args.toxicity_text_col,
            args.toxicity_label_col,
            args.toxicity_language,
            args.toxicity_toxic_label,
            clean_label,
            args.toxicity_skip_clean,
            toxicity_max_rows,
        )

    if not spam_df.empty:
        if args.spam_max_rows <= 0:
            spam_max_rows = len(hebrew_df)
        else:
            spam_max_rows = args.spam_max_rows
        if len(spam_df) > spam_max_rows:
            spam_df = spam_df.sample(n=spam_max_rows, random_state=args.seed)

    df = pd.concat([hebrew_df, spam_df, toxicity_df], ignore_index=True)
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip()
    if args.label_map:
        label_map = load_label_map(Path(args.label_map))
        df["label"] = df["label"].map(lambda label: label_map.get(label, label))

    label_counts = df["label"].value_counts()
    rare_labels = label_counts[label_counts < args.min_class_count].index.tolist()
    if rare_labels:
        if args.rare_strategy == "drop":
            df = df[~df["label"].isin(rare_labels)]
        elif args.rare_strategy == "merge":
            df.loc[df["label"].isin(rare_labels), "label"] = args.rare_label
        elif args.rare_strategy == "keep":
            pass

    label_counts = df["label"].value_counts()

    labels = sorted(df["label"].unique())
    label2id: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}
    id2label: Dict[int, str] = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    stratify = df["label_id"] if label_counts.min() >= 2 else None
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )
    if args.oversample:
        max_count = train_df["label"].value_counts().max()
        oversampled_parts = []
        for label, group in train_df.groupby("label"):
            if len(group) < max_count:
                group = group.sample(n=max_count, replace=True, random_state=args.seed)
            oversampled_parts.append(group)
        train_df = pd.concat(oversampled_parts, ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    class_weights = None
    if args.class_weighting:
        train_label_counts = train_df["label"].value_counts()
        total = float(train_label_counts.sum())
        weights = [total / (len(labels) * train_label_counts[label]) for label in labels]
        class_weights = torch.tensor(weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    eval_dataset = TextDataset(
        eval_df["text"].tolist(),
        eval_df["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = torch.cuda.is_available()
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": args.metric_for_best,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "seed": args.seed,
        "fp16": use_fp16,
        "logging_steps": 50,
        "label_smoothing_factor": args.label_smoothing,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" not in params and "eval_strategy" in params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    training_kwargs = {key: value for key, value in training_kwargs.items() if key in params}
    training_args = TrainingArguments(**training_kwargs)

    trainer_class = WeightedTrainer if args.class_weighting else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "compute_metrics": compute_metrics,
    }
    if args.class_weighting:
        trainer_kwargs["class_weights"] = class_weights
    trainer = trainer_class(**trainer_kwargs)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    finally:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
