import torch
from transformers import pipeline
from inference.constants import SENTIMENT_MAP
from sklearn.metrics import accuracy_score, classification_report



roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0,
    model_kwargs={"torch_dtype": torch.float16}
)

ROBERTA_LABEL_MAP = { #label mapping for specific roberta model
    "LABEL_0": 0,  # Negative
    "LABEL_1": 2,  # Neutral 
    "LABEL_2": 1   # Positive
}

def predict_roberta(text: str):
    result = roberta(text)[0]
    label_str = result["label"]  #label is the sentiment class predicted by the model; e.g. LABEL_0, LABEL_1, LABEL_2                                            
    label_id = ROBERTA_LABEL_MAP[label_str]

    return {
        "label_id": int(label_id),
        "label": SENTIMENT_MAP[label_id], #this is a different label to the label in the result dict and label str
        "score": float(result["score"]),
        "model": "roberta"
    }

def evaluate_roberta_on_dataset(X_test, y_test):
    print("Evaluating RoBERTa model...")
    y_pred = []
    batch_size = 32 #we use batch processing since Transformers is a large model and processing one text at a time would be inefficient and slow

    texts = list(X_test) # xtest to list for batch processing

    for i in range(0, len(texts), batch_size): #loop through texts in batches 
        batch = texts[i:i+batch_size] #get batch of texts; i is the starting index and i+batch_size is the ending index for the batch
        results = roberta(batch, truncation=True, max_length=512) #ensure max length does not exceed 512 tokens for roberta or it might crash
        
        for r in results:
            y_pred.append(ROBERTA_LABEL_MAP[r["label"]]) #predicted label value is appended to y_pred list using the ROBERTA_LABEL_MAP to convert from label string to numerical value e.g. LABEL_0 to 0
        
        if (i + len(batch)) % 1000 < 32: # Print once every 1000 processed for each batch to avoid printing every batch and keep console clean
            print(f"Processed {i + len(batch)} / {len(texts)}")


    print("RoBERTa Accuracy:", accuracy_score(y_test, y_pred)) #compare the actual test results and the predicted results for evaluation
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive", "Neutral"]))
    
    return accuracy_score(y_test, y_pred)



    