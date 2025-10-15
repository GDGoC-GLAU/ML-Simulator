import os

MODEL_DIR = "models"

def save_uploaded_model(file):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, file.name), "wb") as f:
        f.write(file.getbuffer())
