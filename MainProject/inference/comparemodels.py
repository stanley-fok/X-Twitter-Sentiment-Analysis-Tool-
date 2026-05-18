from inference.logreg_infer import predict_logreg
from inference.naivebayes_infer import predict_naive_bayes
from inference.vader_infer import predict_vader
from inference.distilbert_infer import predict_distilbert
from inference.roberta_infer import predict_roberta
from inference.constants import SENTIMENT_MAP
#this file goes to app.py; in python flask

def normalize_result(result, model_name): #to distinguish and standardise the format/value of rule based and ML based sentiment analysis models for consistency
#transforming disparate output formats—such as word-list scores, raw probabilities, or categorical labels—into a unified, standardized scale
        return {
            "model": model_name,
            "label_id": result["label_id"],
            "label": result["label"],
            "score": float(result["score"])  #keep at a value greater than or equal to 0 for top pos/neg posts feature and fallback value
        }
   


def compare_models(text: str): #store results of models with associated value for each sentiment model type as a list
    results = [] #all values computed come from each infer file 

    results.append(normalize_result(
        predict_logreg(text),
        "Logistic Regression"
    ))

    results.append(normalize_result(
        predict_naive_bayes(text),
        "Naive Bayes"
    ))

    results.append(normalize_result(
        predict_vader(text),
        "VADER"
    ))

    results.append(normalize_result(
        predict_distilbert(text),
        "DistilBERT"
    ))

    results.append(normalize_result(
        predict_roberta(text),
        "RoBERTa"
    ))

    return results




