import json
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

nltk.download("vader_lexicon")

tqdm.pandas()

with open("dataset_with_thumbnail_features.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("Dataset loaded:", len(df))

sia = SentimentIntensityAnalyzer()


def get_sentiment(text):

    if not isinstance(text, str) or text.strip() == "":
        return 0

    scores = sia.polarity_scores(text)

    return scores["compound"]


print("Extracting title sentiment...")
df["titleSentiment"] = df["title"].progress_apply(get_sentiment)

print("Extracting description sentiment...")
df["descriptionSentiment"] = df["description"].progress_apply(get_sentiment)


print("\nExample sentiment values:")
print(df[["titleSentiment", "descriptionSentiment"]].head())


output_file = "dataset_with_text_features.json"

df.to_json(output_file, orient="records", indent=2)

print("\nSentiment extraction complete")
print("Saved:", output_file)