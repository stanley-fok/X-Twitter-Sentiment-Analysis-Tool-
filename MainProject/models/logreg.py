# models/logreg.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
#we do not actually run this file 
#this file is the model itself for multinomial logistic regression focusing on tf idf implementation that we run in the location file trainmodel.py 
#it automatically becomes multinomial when detects a third input

def train_logistic_regression( #these variables and its values are from trainmodel.py
    X_train, X_test, y_train, y_test, model_path, vectorizer_path):
    # Vectorisation analyses word importance relative to corpus or set of tokenised text (feature extraction)
    print("Vectorising text...")
    vectorizer = TfidfVectorizer(
        max_features=10000, #vocabulary size limit
        ngram_range=(1, 2) #captures unigrams and bigrams e.g. (machine, learning, machine learning)
    )
#X is input, y is output labels
    X_train_vec = vectorizer.fit_transform(X_train) #creates dictionary of words from x train then converts words from text to numerical value; learns and applied tf idf
    X_test_vec = vectorizer.transform(X_test) #converts text tested using text learned from training vec which is converted to numerical values; applies tf idf learned from training to test data
    #we avoid fitting in test values to show consistency in the data we trained to what we are testing
    

    # Model training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000) #log reg is an iterative optimisation algorithm - converges to best solution each iteration
    model.fit(X_train_vec, y_train) # training; learn the relationship between input and labels; important words vs sentiment labels

    # Evaluation
    print("Evaluating Logistic Regression model...")
    y_pred = model.predict(X_test_vec) #uses trained model to predict test portions or X test vec which are unseen data by the model
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred)) #compare the actual test results and the predicted results for evaluation
    print(classification_report(y_test, y_pred)) #results

    # Save artefacts
    print("Saving Logistic Regression artifacts...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Logistic Regression training complete.\n")

    return model
