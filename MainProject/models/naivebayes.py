
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


#we do not actually run this file
#this file is the model itself for naive bayes that we run in the location file trainmodel.py 

def train_naive_bayes(X_train, X_test, y_train, y_test, model_path, vectorizer_path):
    print("Vectorising text...")
    #can use count vectorizer but kept for consistency (feature extraction)
    vectorizer = TfidfVectorizer( #step to ensure words are assigned values to show patterns weighing by importance
        max_features=10000,
        ngram_range=(1, 2) 
    )

    X_train_vec = vectorizer.fit_transform(X_train) #creates dictionary of words from x train then converts words from text to numerical value
    X_test_vec = vectorizer.transform(X_test)  #converts text tested using text learned from training vec which is converted to numerical values
    #we avoid fitting in test values to show consistency in the data we trained to what we are testing

    #.fit() learns the rules applied in this case it is vectorizer; .transform() converts assigned vocabulary text importance to numerical values
    
    print("Training Naive Bayes model...")
    model = MultinomialNB() #not iterative, only estimates probability and count word frequencies

    
    model.fit(X_train_vec, y_train) # train the model; learn the relationship between input and labels; important words vs sentiment labels;


    print("Evaluating Naive Bayes model...")
    y_pred = model.predict(X_test_vec) #uses trained model to predict test portions or X test vec which are unseen data by the model
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred)) #compare the actual test results and the predicted results for evaluation
    print(classification_report(y_test, y_pred)) #results

    print("Saving Naive Bayes artifacts...") #save trained models    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Naive Bayes training complete.\n")

    return model
