import torch
from transformers import pipeline
from inference.constants import SENTIMENT_MAP, LABEL_MAP
from sklearn.metrics import accuracy_score, classification_report

distilbert = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    device=0,
    model_kwargs={"torch_dtype": torch.float16} # Use float16 for faster inference on GPU, and reduced virtual memory usage to prevent crash or errors
)

def predict_distilbert(text: str):
    result = distilbert(text)[0]
    label_str = result["label"].lower() #label is the sentiment class predicted by the model; e.g. negative, positive, neutral. We use .lower() to ensure consistency with the keys in LABEL_MAP which are all lowercase
    label_id = LABEL_MAP[label_str]

    return {
        "label_id": int(label_id),
        "label": SENTIMENT_MAP[label_id],
        "score": float(result["score"]),
        "model": "distilbert"
    }


def evaluate_distilbert_on_dataset(X_test, y_test):
    print("Evaluating DistilBERT model...")
    y_pred = []
    batch_size = 32 #we use batch processing since Transformers is a large model and processing one text at a time would be inefficient and slow
    texts = list(X_test) # xtest to list for batch processing

    for i in range(0, len(texts), batch_size): #loop through texts in batches
        batch = texts[i : i + batch_size] #get batch of texts; i is the starting index and i+batch_size is the ending index for the batch
        results = distilbert(batch) #ensure max length does not exceed 512 tokens for roberta or it might crash
        
        for r in results:
            label_str = r["label"].lower() #keep .lower() for consistency with LABEL_MAP keys
            y_pred.append(LABEL_MAP[label_str]) #predicted label value is appended to y_pred list using the LABEL_MAP to convert from label string to numerical value e.g. negative to 0, positive to 1, neutral to 2
        
        if (i + len(batch)) % 1000 < 32: # Print once every 1000 processed for each batch to avoid printing every batch and keep console clean
             print(f"Processed {i + len(batch)} / {len(texts)}")

    print("DistilBERT Accuracy:", accuracy_score(y_test, y_pred)) 
    
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive", "Neutral"]))
    return accuracy_score(y_test, y_pred)


