from flask import Flask, request, render_template
import pandas as pd
from inference.model_router import predict_sentiment
from inference.comparemodels import compare_models
from collections import Counter
from nltk.corpus import stopwords
import re


# this file is the Python Flask app file which handles the backend of the project
app = Flask(__name__)
# Load most up to date dataset once
df = pd.read_csv("data/processed/processed_tweets_with_topics.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    #set values to None or 0 to begin with before using the app
    prediction = None
    filtered_posts = None
    comparison_results = None
    num_posts = 10
    word_counts = []
    top_positive_posts = []
    top_negative_posts = []
    scored_posts = []
    error_message = None
    search_text = None
    selected_topic = None
    selected_model = None

    sentiment_counts = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }

    TOPIC_LABELS = {
    0: "Conversation",
    1: "Entertainment",
    2: "Personal Life",
    3: "Daily Routine",
    4: "Social Media",
    5: "Finance & Technology",
    6: "Anticipation/Future",
    7: "Sleep",
    8: "Positivity",
    9: "Weather"
   
}
    filtered_df = df.copy()

    if request.method == "POST":
        search_text = request.form.get("tweet")
        selected_model = request.form.get("model") #retrieve model name from routing to model_router.py for prediction
        selected_topic = request.form.get("topic")

        if not search_text or search_text.strip() == "":
            error_message = "Please enter a keyword or hashtag before running sentiment analysis."

        try:
            num_posts = min(int(request.form.get("num_posts", 10)), 200) #analyses 10 posts by default if no filter selected with min of 200 posts
        except (ValueError, TypeError):
            num_posts = 10
        

        try:
            #model prediction section
            if search_text:
                prediction = predict_sentiment(search_text, selected_model)

                #model comparison to prepare displaying the results from backend to frontend
                comparison_results = compare_models(search_text)


                #filtering functionality for frontend 
                if selected_topic and selected_topic.strip() != "": #filters selected topic according to lda topic number and list
                    filtered_df = filtered_df[filtered_df["lda_topic"] == int(selected_topic)]  

                
                if search_text: #if search text exists, must contain the search text in the tweet column of the dataset 
                    filtered_df = filtered_df[
                        filtered_df["tweet"].str.contains(search_text, case=False, na=False)
                    ]
                
              
               
                #total post analysed using random sample from .sample function to ensure fair collection of tweets
                if filtered_df.empty:
                    error_message = "No tweets matched your search criteria!"
                else:
                    filtered_df = filtered_df.sample(
                        n=min(num_posts, len(filtered_df)), #how many rows to sample to ensure it doesnt exceed dataset (num posts selected by user)
                        random_state=42 #ensures consistency in algorithm and results
                    )

                   
                
                    for tweet_text in filtered_df["tweet"]:#using filttered_df 
                        results = compare_models(tweet_text)#uses function to store normalized model results appropriate for the given text tweet (sentiment analysis)
                        
                        # get result for selected model
                        selected_result = next(
                            (r for r in results if r["model"].lower() == selected_model.lower()), #iterates through each model in compare model results
                            None)
                        print("Selected result:", selected_result)

                        if selected_result: #label is from comparemodels.py
                            label = selected_result["label"].lower()
                            score = selected_result.get("score", 0) #0 if score not available
                            sentiment_counts[label] += 1 #then we increment sentiment count label per model for the frontend to display e.g. positive,negative,neutral




                            scored_posts.append({ #used for top positive and negative posts based on score and label for the selected model
                                "text": tweet_text,
                                "label": label,
                                "score": score
                            })        

                    positive_posts = [p for p in scored_posts if p["label"] == "positive" and p["score"] is not None] #score and label corresponds then proceed
                    negative_posts = [p for p in scored_posts if p["label"] == "negative" and p["score"] is not None]

                    top_positive_posts = sorted(
                        positive_posts,
                        key=lambda x: x["score"], #reverse used to attain highest value of positive 
                        reverse=True
                    )[:5]

                    top_negative_posts = sorted(
                        negative_posts,
                        key=lambda x: x["score"], #reverse used to attain highest value of negative 
                        reverse=True 
                    )[:5]

                
                    filtered_posts = filtered_df #final filtered dataset (posts) that is displayed in the frontend after all filters applied and random sampling for num posts applied

                    if not filtered_df.empty: #create to remove stopwords in wordcloud if it is a filtered df that isnt empty.
                        all_text = " ".join(filtered_df["tweet"].astype(str)) #place all text in one string for wordcloud
                        words = re.findall(r'\b\w+\b', all_text.lower()) #use regex to find all words in the text and convert to lowercase for consistency; \b is word boundary, \w is word character, + is one or more, so it finds sequences of word characters as words
                        stop_words = set(stopwords.words('english'))
                        words = [w for w in words if len(w) > 3 and w not in stop_words] #define proper words and remove stop words
                        word_counts = Counter(words).most_common(50) #display top 50 words based on frequency

                   
        except:
            print("System Error")
            
  

    return render_template( #return results to HTML for display
        "index.html",
        input_text = search_text,
        prediction=prediction,
        posts=filtered_posts,
        comparison=comparison_results,
        sentiment_counts=sentiment_counts,
        num_posts=num_posts,
        word_counts=word_counts,
        top_positive_posts=top_positive_posts,
        top_negative_posts=top_negative_posts,
        topic_labels=TOPIC_LABELS,
        error_message=error_message,
        topic=selected_topic,
        model=selected_model
    )

if __name__ == "__main__": 
    app.run(debug=True)

