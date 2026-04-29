import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

np.random.seed(42)

# Load dataset
with open("dataset_preprocessed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)


print("Total dataset entries:", len(df))


# Feature Engineering
df["titleLength"] = df["title"].astype(str).apply(len)
df["descriptionLength"] = df["description"].astype(str).apply(len)
df["hashtagCount"] = df["hashtags"].astype(str).apply(lambda x: len(x.split(";")) if x else 0)
df["tagCount"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")) if x else 0)

df["thumbnailDominantEmotion"] = df["thumbnailDominantEmotion"].fillna("unknown")

df = pd.get_dummies(df, columns=["thumbnailDominantEmotion"], prefix="emotion")

emotion_features = [col for col in df.columns if col.startswith("emotion_")]


# Target Variable
df["view_sub_ratio"] = df["viewCountDay30"] / df["channelSubscriberCount"]
df["viral"] = (df["view_sub_ratio"] >= 1.0).astype(int)

print("\nViral class distribution:")
print(df["viral"].value_counts())
print(df["viral"].value_counts(normalize=True))



# Feature Groups
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


# Training Function

def train_svm(feature_list, label):

    df_clean = df.dropna(subset=feature_list + ["viral"])

    X = df_clean[feature_list]
    y = df_clean["viral"]

    print(f"\nDataset size for {label}: {len(df_clean)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )



    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))


    model = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\n{label}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", acc)
    print("ROC-AUC:", roc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Cross-validation ROC-AUC:", scores.mean())

    return roc, scores.mean(), model, X_test, y_test




# Train Models

meta_auc, meta_cv, meta_model, X_test_meta, y_test_meta = train_svm(
    metadata_features,
    "Metadata SVM"
)

thumb_auc, thumb_cv, thumb_model, X_test_thumb, y_test_thumb = train_svm(
    thumbnail_features,
    "Thumbnail SVM"
)

comb_auc, comb_cv, comb_model, X_test_comb, y_test_comb = train_svm(
    combined_features,
    "Combined SVM"
)


# No Day 1 models

meta_no_day1_auc, meta_no_day1_cv, meta_no_day1_model, X_test_meta_no_day1, y_test_meta_no_day1 = train_svm(
    metadata_no_day1_features,
    "Metadata (No Day 1)"
)

comb_no_day1_auc, comb_no_day1_cv, comb_no_day1_model, X_test_comb_no_day1, y_test_comb_no_day1 = train_svm(
    combined_no_day1_features,
    "Combined (No Day 1)"
)


# Results Comparison

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

print("\nSVM Performance:")
print(results)


# Model Comparison Plot

results.set_index("Model")[["Test ROC-AUC", "CV ROC-AUC"]].plot(kind="bar", figsize=(10,6))
plt.title("SVM Model Comparison (With and Without Day 1)")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ROC Curve Comparison
fig, ax = plt.subplots(figsize=(8,6))

RocCurveDisplay.from_estimator(
    meta_model,
    X_test_meta,
    y_test_meta,
    name=f"Metadata (AUC={meta_auc:.2f})",
    ax=ax
)

RocCurveDisplay.from_estimator(
    thumb_model,
    X_test_thumb,
    y_test_thumb,
    name=f"Thumbnail (AUC={thumb_auc:.2f})",
    ax=ax
)

RocCurveDisplay.from_estimator(
    comb_model,
    X_test_comb,
    y_test_comb,
    name=f"Combined (AUC={comb_auc:.2f})",
    ax=ax
)

ax.set_title("SVM ROC Curves")
plt.show()


joblib.dump(meta_model, "models/svm_metadata_model.pkl")
joblib.dump(thumb_model, "models/svm_thumbnail_model.pkl")
joblib.dump(comb_model, "models/svm_combined_model.pkl")
joblib.dump(meta_no_day1_model, "models/svm_metadata_no_day1.pkl")
joblib.dump(comb_no_day1_model, "models/svm_combined_no_day1.pkl")

#print("SVM kernel:", model.named_steps["svm"].kernel)