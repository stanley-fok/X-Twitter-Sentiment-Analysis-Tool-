# variables to map for different formats of output for the models e.g. rule based models use strings while machine learning based uses integers
# goes to vader, distilbert, roberta, 
LABEL_MAP = {
    "negative": 0,
    "positive": 1,
    "neutral": 2
}

SENTIMENT_MAP = {
    0: "negative",
    1: "positive",
    2: "neutral"
}
