import pandas as pd
import os
import zipfile

#this file handles the downloading and processing of dataset

API_FILE = "../Finalyearproject/MainProject/twitter_dataset.csv" #file downloaded from Kaggle
PROCESSED_FILE = "data/processed/processed_tweets.csv" #file that is cleaned and processed
LOCAL_FILE = "twitter_dataset_offline.csv" #file that is downloaded and used from local machine 



def loaddatasets():
    try:
        # download from Kaggle
        os.system(
            'kaggle datasets download ' #kaggle CLI (Command line interface)
            '-d prkhrawsthi/twitter-sentiment-dataset-3-million-labelled-rows ' #dataset name from Kaggle
            '-f twitter_dataset.csv -p .' #download file name and path to current directory 
        )

        if os.path.exists("twitter_dataset.csv.zip"): #verify if file exists in project directory or is downloaded locally
            with zipfile.ZipFile(
                "twitter_dataset.csv.zip", 'r' #extract zip file and use .csv file
            ) as zip_ref:
                zip_ref.extractall(".")

        df = pd.read_csv(API_FILE)

    except Exception:
        df = pd.read_csv(LOCAL_FILE)


    if 'tweet' not in df.columns: #check most important column 
        raise ValueError("Dataset must contain 'tweet' column")

    return df

df = loaddatasets()
    
def clean_data(df):
    

    # keep only required columns
    df = df[['tweet', 'sentiment']]

    # drop missing tweets
    df = df.dropna(subset=['tweet', "sentiment"])

    # remove duplicate tweets
    df = df.drop_duplicates(subset=['tweet'])

    # strip whitespace
    df['tweet'] = df['tweet'].astype(str).str.strip()

    # remove extremely short tweets by selecting str len higher than 3
    df = df[df['tweet'].str.len() > 3]



    return df


df = clean_data(df)



# save processed dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_FILE, index=False)


#check rows

print((df['sentiment'] == 0).sum())


print((df['sentiment'] == 1).sum())


print((df['sentiment'] == 2).sum())


# test
if __name__ == "__main__":
    df = loaddatasets()
    df = clean_data(df)
    print(df.head())
    print(df.shape)
