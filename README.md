# 🤖 ML Model Simulator

A Streamlit-based platform to explore basic ML models interactively.

## Features
- 📈 Linear Regression with adjustable parameters
- 📊 Logistic Regression with decision boundary visualization
- ✅ Easy to extend with new models

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

```

## Running with Docker
```bash
docker build -t ml-simulator .
docker run -p 8501:8501 ml-simulator
```
Then open your browser to `http://localhost:8501`.
