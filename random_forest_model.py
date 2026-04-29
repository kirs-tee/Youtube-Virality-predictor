import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,RocCurveDisplay,accuracy_score,f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


np.random.seed(42)


# Load dataset
with open("dataset_preprocessed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("Total dataset entries:", len(df))


# Metadata feature engineering
df["titleLength"] = df["title"].astype(str).apply(len)
df["descriptionLength"] = df["description"].astype(str).apply(len)
df["hashtagCount"] = df["hashtags"].astype(str).apply(lambda x: len(x.split(";")) if x else 0)
df["tagCount"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")) if x else 0)


# Make emotion column numerical features
df["thumbnailDominantEmotion"] = df["thumbnailDominantEmotion"].fillna("unknown")

df = pd.get_dummies(df, columns=["thumbnailDominantEmotion"], prefix="emotion")

emotion_features = [col for col in df.columns if col.startswith("emotion_")]




# Define viral target
df["view_sub_ratio"] = df["viewCountDay30"] / df["channelSubscriberCount"]
df["viral"] = (df["view_sub_ratio"] >= 1.0).astype(int)

print("\nViral class distribution:")
print(df["viral"].value_counts())
print(df["viral"].value_counts(normalize=True))


# Feature groups
metadata_features = [
    "channelSubscriberCount",
    "durationSeconds",
    "viewCountDay1",
    "likeCountDay1",
    "commentCountDay1",
    "titleLength",
    "descriptionLength",
    "hashtagCount",
    "tagCount",
    "titleSentiment",
    "descriptionSentiment"
]
thumbnail_features = [
    "thumbnailBrightness",
    "thumbnailContrast",
    "thumbnailEdgeDensity",
    "thumbnailHasFace",
    "thumbnailFaceCount",
    "thumbnailLargestFaceRatio",
    "thumbnailHasPerson",
    "thumbnailPersonCount",
    "thumbnailAvgSaturation",
    "thumbnailHasText",
    "thumbnailWordCount",
    "thumbnailTextRedRatio",
    "thumbnailHasArrow"
] + emotion_features

combined_features = metadata_features + thumbnail_features

# Create feature lists without Day 1 metrics
metadata_no_day1_features = [
    "channelSubscriberCount",
    "durationSeconds",
    "titleLength",
    "descriptionLength",
    "hashtagCount",
    "tagCount",
    "titleSentiment",
    "descriptionSentiment"
]

combined_no_day1_features = metadata_no_day1_features + thumbnail_features


# Training function
def train_random_forest(feature_list, label):

    df_clean = df.dropna(subset=feature_list + ["viral"])
    X = df_clean[feature_list]
    y = df_clean["viral"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]

    threshold = 0.25  # try 0.2–0.35
    y_pred = (y_prob > threshold).astype(int)

    y_prob = model.predict_proba(X_test)[:,1]

    roc = roc_auc_score(y_test, y_prob)

    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    print(f"\n{label}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Cross-validation ROC-AUC:", scores.mean())

    return roc, scores.mean(), model, X_test, y_test




# Train models - three types to see effect of metadata, thumbnails and them both combines
meta_auc, meta_cv, meta_model, X_test_meta, y_test_meta = train_random_forest(
    metadata_features,
    "Metadata Model"
)

thumb_auc, thumb_cv, thumb_model, X_test_thumb, y_test_thumb = train_random_forest(
    thumbnail_features,
    "Thumbnail Model"
)

comb_auc, comb_cv, comb_model, X_test_comb, y_test_comb = train_random_forest(
    combined_features,
    "Combined Model"
)
# SAVE TEST DATA FOR MODEL COMPARISON
joblib.dump((X_test_comb, y_test_comb), "models/test_data_combined.pkl")

#no day 1
meta_no_day1_auc, meta_no_day1_cv, meta_no_day1_model, X_test_meta_no_day1, y_test_meta_no_day1 = train_random_forest(
    metadata_no_day1_features,
    "Metadata (No Day 1) Model"
)

comb_no_day1_auc, comb_no_day1_cv, comb_no_day1_model, X_test_comb_no_day1, y_test_comb_no_day1 = train_random_forest(
    combined_no_day1_features,
    "Combined (No Day 1) Model"
)

# SAVE TEST DATA FOR NO DAY 1 MODELS
joblib.dump((X_test_comb_no_day1, y_test_comb_no_day1), "models/test_data_no_day1.pkl")

# MODEL COMPARISON TABLE

results = pd.DataFrame({
    "Model": [
        "Metadata",
        "Thumbnail",
        "Combined",
        "Metadata (No Day 1)",
        "Combined (No Day 1)"
    ],
    "Test ROC-AUC": [
        meta_auc,
        thumb_auc,
        comb_auc,
        meta_no_day1_auc,
        comb_no_day1_auc
    ],
    "CV ROC-AUC": [
        meta_cv,
        thumb_cv,
        comb_cv,
        meta_no_day1_cv,
        comb_no_day1_cv
    ]
})

print("\nRandom Forest Performance:")
print(results)


# MODEL COMPARISON PLOT
results.set_index("Model")[["Test ROC-AUC","CV ROC-AUC"]].plot(kind="bar", figsize=(10,6))

plt.title("Random Forest Model Comparison")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ROC CURVES
fig, ax = plt.subplots()

RocCurveDisplay.from_estimator(meta_model, X_test_meta, y_test_meta, name="Metadata RF", ax=ax)
RocCurveDisplay.from_estimator(thumb_model, X_test_thumb, y_test_thumb, name="Thumbnail RF", ax=ax)
RocCurveDisplay.from_estimator(comb_model, X_test_comb, y_test_comb, name="Combined RF", ax=ax)

ax.set_title("Random Forest ROC Curves")

plt.show()


# FEATURE IMPORTANCE
rf_model = comb_model.named_steps["rf"]

importance_df = pd.DataFrame({
    "feature": combined_features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Feature Importance (Combined Model):")
print(importance_df.head(10))

plt.figure(figsize=(8,6))
top_15 = importance_df.head(15)

plt.barh(top_15["feature"], top_15["importance"])
plt.gca().invert_yaxis()

plt.title("Top 15 Features (With Day 1 Metrics)")
plt.tight_layout()

plt.show()


# FEATURE IMPORTANCE (NO DAY 1)
rf_model = comb_no_day1_model.named_steps["rf"]

importance_no_day1_df = pd.DataFrame({
    "feature": combined_no_day1_features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)
print("\nTop Feature Importance (Combined No Day 1 Model):")
print(importance_no_day1_df.head(10))

plt.figure(figsize=(8,6))

top_15_no_day1 = importance_no_day1_df.head(15)

plt.barh(top_15_no_day1["feature"], top_15_no_day1["importance"], color="orange")
plt.gca().invert_yaxis()

plt.title("Top 15 Features (WITHOUT Day 1 Metrics)")
plt.tight_layout()

plt.show()


joblib.dump(meta_model, "models/rf_metadata_model.pkl")
joblib.dump(thumb_model, "models/rf_thumbnail_model.pkl")
joblib.dump(comb_model, "models/rf_combined_model.pkl")
joblib.dump(meta_no_day1_model, "models/rf_metadata_no_day1.pkl")
joblib.dump(comb_no_day1_model, "models/rf_combined_no_day1.pkl")