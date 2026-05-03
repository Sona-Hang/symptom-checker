# 🇳🇵 AI Health Awareness Assistant (Nepal)

A lightweight AI web app designed to help users understand symptoms and decide when to seek medical care.

## Why I built this
Access to healthcare can be limited or delayed in many parts of Nepal.  
This project focuses on **early awareness and basic triage**, not diagnosis.

## What it does
- Symptom-based input (simple UI)
- Suggests possible conditions (top matches)
- Shows confidence levels
- Explains which symptoms influenced the result
- Provides urgency guidance:
  - when to monitor
  - when to visit a clinic
  - when to seek immediate care

## Health focus
Covers common conditions in Nepal:
- Flu, dengue, typhoid
- Tuberculosis (early indicators)
- Stomach infections
- Allergies and minor infections

## Tech stack
- Python, Streamlit
- scikit-learn (ML model)
- Modular structure for training + prediction

## Project structure
- `app.py` — frontend (Streamlit)
- `train_model.py` — model training
- `health_assistant/` — core logic
- `model.pkl` — trained model
- `dataset.csv` — symptom dataset

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py