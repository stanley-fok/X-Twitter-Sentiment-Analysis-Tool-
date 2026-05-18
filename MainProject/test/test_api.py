from pydoc import html
import pytest
from app import app  # import your Flask app
from inference.comparemodels import normalize_result  

@pytest.fixture #make reusable for tests
def client():
    app.config['TESTING'] = True   #set testing mode to True for Flask app
    with app.test_client() as client: #simulates a user interacting with the Flask app without running the server
        yield client


def test_prediction_route(client):
    response = client.post("/", data={ #tests route exists and can handle POST request with form data
        "tweet": "This is a test tweet for sentiment analysis.",    #sample filter inputs test
        "model": "RoBERTa",
        "num_posts": 5
    })

    print(response.data)
    
    assert response.status_code == 200  #tests server error; 200 if it is successful else not
    assert b"Predicted Sentiment" in response.data  #tests keyword "Predicted Sentiment" exists in ("/") http route 
   
    assert b"sentimentChart" in response.data #test sentimentchart id exists
    assert b"wordCloud" in response.data #tests wordcloud id exists

    assert b"Top 5 Positive Posts" in response.data #tests top 5 posts exists
    assert b"Top 5 Negative Posts" in response.data


def test_normalize_result_dict(): #unit test
    result = {"label": "Positive", "score": 0.9} 
    output = normalize_result(result, "RoBERTa")

    assert output["label"] == "Positive" #test if the outputs are correct as inputted (format)
    assert output["score"] == 0.9


