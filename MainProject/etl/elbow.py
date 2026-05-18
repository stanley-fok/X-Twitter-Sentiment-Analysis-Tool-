import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
from tqdm import tqdm

from topicmodel import preprocess 

tqdm.pandas()

def run_elbow_method(input_path):
    print("Loading sample for testing...")
    # used same test size sample to keep consistency 
    df = pd.read_csv(input_path).sample(n=622958, random_state=42)
    
    print("Preprocessing tweets (this may take a minute)...") #preprocessing to ensure consistency in data
    df["tokens"] = df["tweet"].progress_apply(preprocess) #apply preprocessing function
    df = df[df["tokens"].map(len) > 0] # remove empty token list and keeps rows with token list length more than 0
    
    dictionary = corpora.Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(text) for text in df["tokens"]]
    texts = df["tokens"].tolist()

    # Define the range of topics to test 
    topic_range = range(5, 31, 5) 
    coherence_scores = []

    for k in topic_range:
        print(f"Testing K={k}...")
        model = LdaMulticore(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=k, 
            passes=2, 
            workers=3, 
            random_state=42
        )
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(topic_range, coherence_scores, marker='o', color='b')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Elbow Method for Number of Topics")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_elbow_method("data/processed/processed_tweets.csv")