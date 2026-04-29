import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Added for Heatmap
from imblearn.over_sampling import SMOTE # Added for SMOTE Graph

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_curve,
    precision_recall_curve, auc # Added for PR Curve
)
from sklearn.model_selection import train_test_split


# LOAD DATASET & TARGET

with open("dataset_preprocessed.json", "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)

df["titleLength"] = df["title"].astype(str).apply(len)
df["descriptionLength"] = df["description"].astype(str).apply(len)
df["hashtagCount"] = df["hashtags"].astype(str).apply(lambda x: len(x.split(";")) if x else 0)
df["tagCount"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")) if x else 0)

df["thumbnailDominantEmotion"] = df["thumbnailDominantEmotion"].fillna("unknown")
df = pd.get_dummies(df, columns=["thumbnailDominantEmotion"], prefix="emotion")
emotion_features = [col for col in df.columns if col.startswith("emotion_")]

df["view_sub_ratio"] = df["viewCountDay30"] / df["channelSubscriberCount"]
df["viral"] = (df["view_sub_ratio"] >= 1.0).astype(int)

#  FEATURE GROUPS

metadata_features = [
    "channelSubscriberCount","durationSeconds","viewCountDay1","likeCountDay1",
    "commentCountDay1","titleLength","descriptionLength","hashtagCount",
    "tagCount","titleSentiment","descriptionSentiment"
]
thumbnail_features = [
    "thumbnailBrightness","thumbnailContrast","thumbnailEdgeDensity",
    "thumbnailHasFace","thumbnailFaceCount","thumbnailLargestFaceRatio",
    "thumbnailHasPerson","thumbnailPersonCount","thumbnailAvgSaturation",
    "thumbnailHasText","thumbnailWordCount","thumbnailTextRedRatio",
    "thumbnailHasArrow"
] + emotion_features

combined_features = metadata_features + thumbnail_features

metadata_no_day1 = [
    "channelSubscriberCount","durationSeconds","titleLength","descriptionLength",
    "hashtagCount","tagCount","titleSentiment","descriptionSentiment"
]
combined_no_day1 = metadata_no_day1 + thumbnail_features

feature_sets = {
    "Metadata": metadata_features,
    "Thumbnail": thumbnail_features,
    "Combined": combined_features,
    "Metadata (No Day 1)": metadata_no_day1,
    "Combined (No Day 1)": combined_no_day1
}


# DATASET SNAPSHOT FOR APPENDIX

pd.set_option('display.max_columns', None)
print("\n=== FINAL DATASET SNAPSHOT (FIRST 5 ROWS) ===")
snapshot_cols = combined_features + ["viral"]
print(df[snapshot_cols].head())
pd.reset_option('display.max_columns')


#LOAD ALL 20 MODELS

models = {
    "Random Forest": {
        "Metadata": joblib.load("models/rf_metadata_model.pkl"),
        "Thumbnail": joblib.load("models/rf_thumbnail_model.pkl"),
        "Combined": joblib.load("models/rf_combined_model.pkl"),
        "Metadata (No Day 1)": joblib.load("models/rf_metadata_no_day1.pkl"),
        "Combined (No Day 1)": joblib.load("models/rf_combined_no_day1.pkl")
    },
    "XGBoost": {
        "Metadata": joblib.load("models/xgb_metadata_model.pkl"),
        "Thumbnail": joblib.load("models/xgb_thumbnail_model.pkl"),
        "Combined": joblib.load("models/xgb_combined_model.pkl"),
        "Metadata (No Day 1)": joblib.load("models/xgb_metadata_no_day1.pkl"),
        "Combined (No Day 1)": joblib.load("models/xgb_combined_no_day1.pkl")
    },
    "SVM": {
        "Metadata": joblib.load("models/svm_metadata_model.pkl"),
        "Thumbnail": joblib.load("models/svm_thumbnail_model.pkl"),
        "Combined": joblib.load("models/svm_combined_model.pkl"),
        "Metadata (No Day 1)": joblib.load("models/svm_metadata_no_day1.pkl"),
        "Combined (No Day 1)": joblib.load("models/svm_combined_no_day1.pkl")
    },
    "Logistic Regression": {
        "Metadata": joblib.load("models/lr_metadata_model.pkl"),
        "Thumbnail": joblib.load("models/lr_thumbnail_model.pkl"),
        "Combined": joblib.load("models/lr_combined_model.pkl"),
        "Metadata (No Day 1)": joblib.load("models/lr_metadata_no_day1.pkl"),
        "Combined (No Day 1)": joblib.load("models/lr_combined_no_day1.pkl")
    }
}


# EVALUATION LOOP

results = []
roc_plot_data = {}
pr_plot_data = {} # Added to store Precision-Recall curve data

for algo_name, configs in models.items():
    for feature_name, model in configs.items():
        
        # Isolate the exact columns this specific model needs
        features = feature_sets[feature_name]
        df_clean = df.dropna(subset=features + ["viral"])

        X = df_clean[features]
        y = df_clean["viral"]

        # Split identically to training to prevent data leakage
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        # Store for the table & bar chart
        results.append({
            "Algorithm": algo_name,
            "Feature Type": feature_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc
        })

        # Store for the ROC curve graph
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        label_name = f"{algo_name} ({feature_name})"
        roc_plot_data[label_name] = (fpr, tpr, roc)
        
        # Store for the Precision-Recall curve graph
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        pr_plot_data[label_name] = (recall_vals, precision_vals, pr_auc)


# PRINT CONSOLE TABLE

results_df = pd.DataFrame(results).sort_values(by=["Feature Type", "ROC-AUC"], ascending=[True, False])
print("\n=== COMPREHENSIVE MODEL COMPARISON ===")
print(results_df.to_string(index=False))


# ROUPED BAR CHART

plt.rcParams.update({'font.size': 14}) 

pivot = results_df.pivot_table(
    index="Feature Type",
    columns="Algorithm",
    values="ROC-AUC"
)

desired_order = ["Metadata", "Thumbnail", "Combined", "Metadata (No Day 1)", "Combined (No Day 1)"]
pivot = pivot.reindex(desired_order)

ax = pivot.plot(kind="bar", figsize=(14, 8), width=0.75)
plt.title("Model Comparison Across All Feature Modalities", fontsize=14, pad=20)
plt.ylabel("ROC-AUC Score", fontsize=16)
plt.xlabel("Feature Type", fontsize=16)
plt.xticks(rotation=35, ha="right", fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title="Algorithm", title_fontsize='15', fontsize='13', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.show()


#   ROC CURVE

plt.figure(figsize=(14, 10))

for label, (fpr, tpr, auc_val) in roc_plot_data.items():
    plt.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC = {auc_val:.2f})")

plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Baseline')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Comprehensive ROC Curves for All Modalities', fontsize=16)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#  PRECISION-RECALL CURVE

plt.figure(figsize=(14, 10))

for label, (rec_vals, prec_vals, pr_auc_val) in pr_plot_data.items():
    plt.plot(rec_vals, prec_vals, lw=1.5, label=f"{label} (PR-AUC = {pr_auc_val:.2f})")

# Baseline for PR curve is the ratio of positive class (viral videos) in the test set
baseline = sum(y_test) / len(y_test)
plt.plot([0, 1], [baseline, baseline], color='gray', lw=1.5, linestyle='--', label=f'Baseline ({baseline:.3f})')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall (True Positive Rate)', fontsize=12)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
plt.title('Precision-Recall Curves (Addressing Class Imbalance)', fontsize=16)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# FEATURE CORRELATION HEATMAP

plt.figure(figsize=(16, 12))
# Calculate correlation matrix using only the combined feature set
corr_matrix = df[combined_features].corr()

# Create a mask to only show the bottom triangle (since the top is a mirror image)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, square=True)
plt.title('Multimodal Feature Correlation Heatmap', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# SMOTE DISTRIBUTION BAR CHART

# Grab the clean data for the Combined feature set
df_smote = df.dropna(subset=combined_features + ["viral"])
X_sm = df_smote[combined_features]
y_sm = df_smote["viral"]

# Apply SMOTE just to generate the "After" numbers for the graph
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_sm, y_sm)

before_counts = y_sm.value_counts().sort_index()
after_counts = y_res.value_counts().sort_index()

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Left Chart: Before
bars_before = ax[0].bar(["Non-Viral (0)", "Viral (1)"], before_counts.values, color=['#3498db', '#e74c3c'])
ax[0].set_title("Original Dataset (Severe Imbalance)", fontsize=16, pad=15)
ax[0].set_ylabel("Number of Videos", fontsize=14)
ax[0].tick_params(axis='both', labelsize=12)
ax[0].bar_label(bars_before, padding=3, fontsize=12)

# Right Chart: After
bars_after = ax[1].bar(["Non-Viral (0)", "Viral (1)"], after_counts.values, color=['#3498db', '#e74c3c'])
ax[1].set_title("After SMOTE Balancing Layer", fontsize=16, pad=15)
ax[1].set_ylabel("Number of Videos", fontsize=14)
ax[1].tick_params(axis='both', labelsize=12)
ax[1].bar_label(bars_after, padding=3, fontsize=12)

plt.suptitle("Resolving the Accuracy Paradox: Class Distribution", fontsize=22, y=1.05)
plt.tight_layout()
plt.show()