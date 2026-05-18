from inference.logreg_infer import predict_logreg
from inference.naivebayes_infer import predict_naive_bayes
from inference.vader_infer import predict_vader
from inference.roberta_infer import predict_roberta
from inference.distilbert_infer import predict_distilbert
#goes to app.py; in python flask
#to route and enable model the user selects
def predict_sentiment(text: str, model_name: str):
    if model_name == "Logistic Regression":
        return predict_logreg(text)

    elif model_name == "Naive Bayes":
        return predict_naive_bayes(text)

    elif model_name == "VADER":
        return predict_vader(text)

    elif model_name == "RoBERTa":
        return predict_roberta(text)

    elif model_name == "DistilBERT":
        return predict_distilbert(text)

    else:
        raise ValueError("Unknown model selected")




