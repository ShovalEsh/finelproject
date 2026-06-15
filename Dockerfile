FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY risk_analyzer.py .
COPY url_risk.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]