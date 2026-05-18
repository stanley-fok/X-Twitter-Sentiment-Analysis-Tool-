import pandas as pd
import os
from sklearn.model_selection import train_test_split
from inference.distilbert_infer import evaluate_distilbert_on_dataset
from inference.vader_infer import evaluate_vader_on_dataset
from models.logreg import train_logistic_regression
from models.naivebayes import train_naive_bayes
from inference.roberta_infer import evaluate_roberta_on_dataset


#run using python -m etl.executemodel

#this file executes the training of models that are under machine learning and the evaluation of rule based and pretrained models.


INPUT_FILE = "data/processed/processed_tweets_with_topics.csv" #file afer topic modeling
MODEL_DIR = "data/artifacts" #saved model data from models folder


def main():
    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_FILE)

    # check required columns before training
    required_cols = {"tweet", "sentiment"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain {required_cols}")
    

    df = df.dropna(subset=['tweet', "sentiment"])


    
    X = df["tweet"] #input features
    y = df["sentiment"].astype(int) #labels; ensure consistent labels 

    # train/test split (static dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, #prevent overfitting by choosing 80/20 split
        random_state=42, #ensures consistency in algorithm and results by fixing the randomness for reproducibility; 42 is standard value
        stratify=y #handles balanced ratio of training and test data for consistent data 
    )


    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Executing Logistic Regression model...") #where we execute the LOG REG function from logreg.py
    train_logistic_regression(
        X_train, X_test,
        y_train, y_test,
        model_path=f"{MODEL_DIR}/logreg.pkl", #modeldir is where the saved trained model from logreg.py is generated 
        vectorizer_path=f"{MODEL_DIR}/vectorizer.pkl"
)


    print("Executing Naive Bayes model...") #where we execute the naive bayes function from naivebayes.py
    train_naive_bayes(
        X_train, X_test,
        y_train, y_test,
        model_path=f"{MODEL_DIR}/naivebayes.pkl",
        vectorizer_path=f"{MODEL_DIR}/tfidf_nb.pkl"
)
    
    print("Model execution complete.")


    #evaluate pretrained and rule based models on same test set for consistency in evaluation and comparison of results
    df = pd.DataFrame({
    "tweet": X_test,
    "sentiment": y_test
})
    
    df_sample = df
    print("Evaluating RoBERTa on same test set...")
    evaluate_roberta_on_dataset(df_sample['tweet'], df_sample['sentiment'])

    print("Evaluating DistilBERT on same test set...")
    evaluate_distilbert_on_dataset(df_sample['tweet'], df_sample['sentiment'])

    print("Evaluating VADER on same test set...")
    evaluate_vader_on_dataset(df_sample['tweet'], df_sample['sentiment'])


if __name__ == "__main__":
    main()

    
    




