from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    # Detect regression or classification
    if len(set(y_true)) > 10 or any(isinstance(y, float) for y in y_true):
        return {
            "r2_score": round(r2_score(y_true, y_pred), 3),
            "mean_squared_error": round(mean_squared_error(y_true, y_pred), 3)
        }
    else:
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": round(acc, 3)}
