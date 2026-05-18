from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from inference.constants import SENTIMENT_MAP

analyzer = SentimentIntensityAnalyzer()

def predict_vader(text: str):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label_id = 1   # Positive 
    elif compound <= -0.05:
        label_id = 0   # Negative 
    else:
        label_id = 2   # Neutral 
    
    return {
        "label_id": int(label_id),
        "label": SENTIMENT_MAP[label_id],
        "score": float(compound),
        "model": "vader"
    }


def evaluate_vader_on_dataset(X_test, y_test):
    print("Evaluating VADER... (Rule-based)")
    
    # 1. One-line prediction using your existing predict_vader function
    y_pred = [predict_vader(text)["label_id"] for text in X_test] #gets the label id for each X_test text using the predict_vader function and stores it in y_pred list

    print("VADER Accuracy:", accuracy_score(y_test, y_pred)) #compare the actual test results and the predicted results for evaluation
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive", "Neutral"])) #results

    return accuracy_score(y_test, y_pred)

