# 🤖 ML Model Simulator

A Streamlit-based platform to explore basic ML models interactively.

## Features

- 📈 Linear Regression with adjustable parameters
- 📊 Logistic Regression with decision boundary visualization
- 📥 Upload your own trained estimators and benchmark them instantly
- ✅ Easy to extend with new models

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Upload and Benchmark Your Own Model

1. Launch the app and open **Upload and Benchmark Your Model** from the sidebar.
2. Select the problem type (classification or regression).
3. Upload a `.pkl` or `.joblib` file containing a scikit-learn compatible estimator.
4. Review the generated metrics (accuracy, confusion matrix, MAE/MSE/RMSE, etc.).
5. Encourage collaborators to upvote or favorite the best submissions.
