# Commands

Quick reference for training and prediction. Replace placeholders in <> with your
actual Hebrew labels (see label_map.json).

## Train HF classifier (AlephBERT)
python train_hf_classifier.py `
  --hebrew-data hebrew_dataset2.csv `
  --hebrew-text-col text `
  --hebrew-label-col risk_type `
  --spam-data spam.csv `
  --spam-text-col v2 `
  --spam-label-col v1 `
  --spam-label "התחזות" `
  --ham-label "תקין" `
  --toxicity-dataset "gravitee-io/textdetox-multilingual-toxicity-dataset" `
  --toxicity-language he `
  --toxicity-toxic-label "איום" `
  --toxicity-skip-clean `
  --toxicity-max-rows 50 `
  --label-map label_map.json `
  --min-class-count 5 `
  --rare-strategy merge `
  --rare-label other `
  --oversample `
  --epochs 12 `
  --batch-size 4 `
  --learning-rate 1e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.1 `
  --grad-accum-steps 2 `
  --metric-for-best accuracy `
  --max-length 384 `
  --label-smoothing 0 `
  --seed 42 `
  --model-name "dicta-il/alephbertgimmel-small" `
  --output-dir hf_finetuned_model

## Train HF classifier (higher accuracy: longer context)
python train_hf_classifier.py `
  --hebrew-data hebrew_dataset2.csv `
  --hebrew-text-col text `
  --hebrew-label-col risk_type `
  --spam-data spam.csv `
  --spam-text-col v2 `
  --spam-label-col v1 `
  --spam-label "התחזות" `
  --ham-label "תקין" `
  --toxicity-dataset "gravitee-io/textdetox-multilingual-toxicity-dataset" `
  --toxicity-language he `
  --toxicity-toxic-label "איום" `
  --toxicity-skip-clean `
  --toxicity-max-rows 50 `
  --label-map label_map.json `
  --min-class-count 5 `
  --rare-strategy merge `
  --rare-label other `
  --oversample `
  --epochs 12 `
  --batch-size 2 `
  --learning-rate 1e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.1 `
  --grad-accum-steps 4 `
  --metric-for-best accuracy `
  --max-length 512 `
  --label-smoothing 0 `
  --seed 42 `
  --model-name "dicta-il/alephbertgimmel-small" `
  --output-dir hf_finetuned_model

## Train HF classifier (class weighting instead of oversampling)
python train_hf_classifier.py `
  --hebrew-data hebrew_dataset2.csv `
  --hebrew-text-col text `
  --hebrew-label-col risk_type `
  --spam-data spam.csv `
  --spam-text-col v2 `
  --spam-label-col v1 `
  --spam-label "התחזות" `
  --ham-label "תקין" `
  --toxicity-dataset "gravitee-io/textdetox-multilingual-toxicity-dataset" `
  --toxicity-language he `
  --toxicity-toxic-label "איום" `
  --toxicity-skip-clean `
  --toxicity-max-rows 50 `
  --label-map label_map.json `
  --min-class-count 5 `
  --rare-strategy merge `
  --rare-label other `
  --class-weighting `
  --epochs 12 `
  --batch-size 4 `
  --learning-rate 1e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.1 `
  --grad-accum-steps 2 `
  --metric-for-best accuracy `
  --max-length 384 `
  --label-smoothing 0 `
  --seed 42 `
  --model-name "dicta-il/alephbertgimmel-small" `
  --output-dir hf_finetuned_model

## Train HF classifier (more toxicity rows)
python train_hf_classifier.py `
  --hebrew-data hebrew_dataset2.csv `
  --hebrew-text-col text `
  --hebrew-label-col risk_type `
  --spam-data spam.csv `
  --spam-text-col v2 `
  --spam-label-col v1 `
  --spam-label "התחזות" `
  --ham-label "תקין" `
  --toxicity-dataset "gravitee-io/textdetox-multilingual-toxicity-dataset" `
  --toxicity-language he `
  --toxicity-toxic-label "איום" `
  --toxicity-skip-clean `
  --toxicity-max-rows 200 `
  --label-map label_map.json `
  --min-class-count 5 `
  --rare-strategy merge `
  --rare-label other `
  --oversample `
  --epochs 12 `
  --batch-size 4 `
  --learning-rate 1e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.1 `
  --grad-accum-steps 2 `
  --metric-for-best accuracy `
  --max-length 384 `
  --label-smoothing 0 `
  --seed 42 `
  --model-name "dicta-il/alephbertgimmel-small" `
  --output-dir hf_finetuned_model

## Train sklearn spam model (optional English spam baseline)
python train_sklearn_spam.py

## Predict on a CSV with the fine-tuned HF model
python main.py hebrew_sentiment_test.csv `
  --text-col text `
  --encoding utf-8-sig `
  --hf-model-path hf_finetuned_model `
  --csv-out hebrew_sentiment_scored.csv

# Examples:
# Spam dataset (latin-1, text column v2):
# python main.py spam.csv --text-col v2 --encoding latin-1 --hf-model-path hf_finetuned_model --csv-out scored.csv
#
# Hebrew dataset (UTF-8 with BOM):
# python main.py hebrew_dataset2.csv --text-col text --encoding utf-8-sig --hf-model-path hf_finetuned_model --csv-out scored.csv


python train_hf_classifier.py --hebrew-data hebrew_dataset2.csv --hebrew-text-col text --hebrew-label-col risk_type --spam-data spam.csv --spam-text-col v2 --spam-label-col v1 --spam-label "התחזות" --ham-label "תקין" --toxicity-dataset "gravitee-io/textdetox-multilingual-toxicity-dataset" --toxicity-language he --toxicity-toxic-label "איום" --toxicity-skip-clean --toxicity-max-rows 50 --label-map label_map.json --min-class-count 5 --rare-strategy merge --rare-label other --oversample --epochs 20 --batch-size 4 --learning-rate 1e-5 --weight-decay 0.01 --warmup-ratio 0.1 --grad-accum-steps 2 --metric-for-best accuracy --max-length 384 --label-smoothing 0 --seed 42 --model-name "dicta-il/alephbertgimmel-small" --output-dir hf_finetuned_model
textdetox/bert-multilingual-toxicity-classifier
dicta-il/alephbertgimmel-small
SinaLab/Offensive-Hebrew