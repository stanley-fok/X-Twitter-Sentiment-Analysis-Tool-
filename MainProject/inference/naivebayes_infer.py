import joblib
import numpy as np
from inference.constants import SENTIMENT_MAP 

MODEL_PATH = "data/artifacts/naivebayes.pkl"
VECTORIZER_PATH = "data/artifacts/tfidf_nb.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_naive_bayes(text: str): # where the model is used for sharing results
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]   # probability of classification
    label_id = int(np.argmax(probs)) # picks largest value
    confidence = float(probs[label_id]) # displays score of value

    return {
        "model": "naive_bayes",
        "label_id": int(label_id),
        "label": SENTIMENT_MAP[label_id],
        "score": float(confidence) #returns score for naivebayes
    }


  
