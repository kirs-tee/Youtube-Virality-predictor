import json
import pandas as pd

# LOAD DATA

with open("dataset_with_text_features.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("Original dataset size:", len(df))

df = df.drop(columns=["thumbnailFilename"])

# FIX ROWS WITH NO VALUES
# Fill missing emotions with "None" and confidence with 0
df["thumbnailDominantEmotion"] = df["thumbnailDominantEmotion"].fillna("None")
df["thumbnailEmotionConfidence"] = df["thumbnailEmotionConfidence"].fillna(0)

# Drop rows where the basic image stats are missing
df = df.dropna(subset=["thumbnailBrightness"])


# HANDLE MISSING VALUES
print("\nMissing values per field:")
print(df.isnull().sum())

print(f"\nTotal entries containing missing values: {df.isnull().any(axis=1).sum()}")

#sentiment analysis none is 0
df["titleSentiment"] = df["titleSentiment"].fillna(0)
df["descriptionSentiment"] = df["descriptionSentiment"].fillna(0)

# Drop standard nulls
df = df.dropna()

print(f"\nTotal entries after deleting missing values: {len(df)}")
# REMOVE HIDDEN LIKES AND DISABLED COMMENTS
# Keep only rows where day 30 likes and comments are greater than 0
initial_size = len(df)
df = df[(df["likeCountDay30"] > 0) & (df["commentCountDay30"] > 0)]

print(f"\nRemoved {initial_size - len(df)} rows with 0 likes or comments.")
print("Dataset size after cleaning:", len(df))

# CREATE TARGET VARIABLE (VIRAL)

df["viral"] = (df["viewCountDay30"] >= 50000).astype(int)

print("\nViral distribution:")
print(df["viral"].value_counts())
print(df["viral"].value_counts(normalize=True))


# METADATA FEATURE ENGINEERING

df["titleLength"] = df["title"].astype(str).apply(len)
df["descriptionLength"] = df["description"].astype(str).apply(len)
df["hashtagCount"] = df["hashtags"].astype(str).apply(
    lambda x: len([h for h in x.split(";") if h.strip() != ""])
)
df["tagCount"] = df["tags"].astype(str).apply(
    lambda x: len([t for t in x.split("|") if t.strip() != ""])
)

# Convert booleans to integers
df["thumbnailHasPerson"] = df["thumbnailHasPerson"].astype(int)
df["thumbnailHasArrow"] = df["thumbnailHasArrow"].astype(int)

# Convert categoryId
df["categoryId"] = pd.to_numeric(df["categoryId"], errors="coerce")



# DATE AND TIME FEATURES
# Convert the string to a proper datetime object
df["publishedAt"] = pd.to_datetime(df["publishedAt"])

# Extract useful time-based features
df["publishHour"] = df["publishedAt"].dt.hour
df["publishDayOfWeek"] = df["publishedAt"].dt.dayofweek # 0 = Monday, 6 = Sunday
df["isWeekend"] = df["publishDayOfWeek"].isin([5, 6]).astype(int)



# SAVE CLEAN DATASET

df.to_json("dataset_preprocessed.json", orient="records", indent=2)

print("\nSaved dataset_preprocessed.json")