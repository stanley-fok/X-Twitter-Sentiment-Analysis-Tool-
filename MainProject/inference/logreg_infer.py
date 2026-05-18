import joblib
import numpy as np
from inference.constants import SENTIMENT_MAP

MODEL_PATH = "data/artifacts/logreg.pkl"
VECTORIZER_PATH = "data/artifacts/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_logreg(text: str):
    X = vectorizer.transform([text])
    # probabilities for each class 
    probs = model.predict_proba(X)[0] #softmax
    label_id = int(np.argmax(probs)) # picks largest value
    confidence = float(probs[label_id]) # displays score of value

    return {
        "model": "logreg",
        "label_id": int(label_id),
        "label": SENTIMENT_MAP[label_id],
        "score": float(confidence) #returns the score for log reg
    }





