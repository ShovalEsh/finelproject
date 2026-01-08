# finelproject

Setup and run instructions for Windows PowerShell.

## Setup

```powershell
cd C:\Users\lubac\vscodeproj\oldies\finelproject
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Hebrew phishing model

```powershell
python train_phishing_model.py
```

This creates `models/hebrew-phishing-model`, which the API loads on startup.

## Run the API

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8001
```

If port 8001 is blocked, pick another port (for example `8002`).

## Test the API (UTF-8 payload)

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8001/analyze `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes((@{ text = "YOUR_HEBREW_TEXT_HERE" } | ConvertTo-Json -Compress)))
```

## Optional: enable zero-shot scoring

Zero-shot adds multilingual labels but is slower and can be noisy for Hebrew.

```powershell
$env:USE_ZERO_SHOT = "true"
uvicorn api:app --reload --host 127.0.0.1 --port 8001
```

## Notes

- For a larger multi-class training pipeline, see `train_hf_classifier.py` and `COMMANDS.md`.
