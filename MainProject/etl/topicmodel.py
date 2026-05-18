import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaMulticore
import os
from tqdm import tqdm


#this file uses LDA machine learning model to breakdown keywords to create subtopics which groups words into similar categories


# setup & downloads for text pre processing

nltk.download('punkt', quiet=True) #punkt is a pre-trained model for tokenization, which is the process of breaking down text into individual words or tokens. 
nltk.download('stopwords', quiet=True) #set of stopwords pre-made

STOP_WORDS = set(stopwords.words('english')) #stopwords are words like as, of, the

tqdm.pandas() # enable progress bar

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) #enable structured messages (metadata) with progress bar

def preprocess(text): #not LDA but needed for LDA to remove unwanted text in the tweet column
    if not isinstance(text, str) or pd.isna(text): # if text not string or is null then return empty list as error handling
        return []
    text = text.lower() #make letters consistent by sicking to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) #remove any website domain text
    text = re.sub(r"[^a-zA-Z]", " ", text) #remove any nonletter (not lower or uppercase alphabets)
    tokens = word_tokenize(text) #splits texts into individual words; punkt 
    return [w for w in tokens if w not in STOP_WORDS and len(w) > 2] #if item in tokens variable is not STOP WORDS e.g. the, is, to and greater than 2, return value



def run_lda(input_path, output_path, num_topics=10):
    # check if file exists first
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    print("Loading data...")
    df = pd.read_csv(input_path)

    # text for progress bar
    print("Preprocessing tweets...")
    # uses a progress bar from tqdm.pandas() while applying preprocess function on tweet to create tokens and column
    df["tokens"] = df["tweet"].progress_apply(preprocess)

    # remove empty token list and keeps rows with token list length more than 0
    df = df[df["tokens"].map(len) > 0]

    print("Building dictionary and corpus...")
    dictionary = corpora.Dictionary(df["tokens"]) #builds vocabulary from column tokens (tokenised tweets); assigning each tokens as a unique ID value
    corpus = [dictionary.doc2bow(text) for text in df["tokens"]] #converts each token into Bag of words format. [(word id, word count)]
    #corpus is the set of tokenised texts in tokens column which is converted to BOW

    # 3. LDA training using Gensim's LdaMulticore
    print(f"Training LDA model with {num_topics} topics...")
    lda_model = LdaMulticore(
        corpus=corpus, #bagofwords data
        id2word=dictionary, #dictionary id values convert back to words for readability
        num_topics=num_topics, #chosen number of topics which is 20 for balance of dataset of 1+million rows
        passes=7, # keep at 7 for balanced speed and efficient computation
        workers=3, # uses multiple CPU cores handling big data
        random_state=42 #ensures consistency in algorithm and results
    )

    # assign dominant topic
    def get_topic(bow): #bow == one tweet bag of words in corpus
        topics = lda_model.get_document_topics(bow) #computes probability tokenised tweet is a certain topic from BOW
        return max(topics, key=lambda x: x[1])[0] #selects the highest probability that the tweet is the topic using max function
    #x is iteration for each topics; topic number is output returned which is the first index of the selected maximum x[1] which is the probability (2nd index) 

    df["lda_topic"] = [get_topic(bow) for bow in corpus] #assign function to column ldatopic


    # save outputs
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    df.to_csv(output_path, index=False)
    dictionary.save("data/models/lda_dictionary.dict")
    lda_model.save("data/models/lda_model.model")

    print("\nSuccess!")
    print(f"Processed dataset saved to: {output_path}")



if __name__ == "__main__":
    run_lda(
        input_path="data/processed/processed_tweets.csv",
        output_path="data/processed/processed_tweets_with_topics.csv",
        num_topics=10
    )
    

