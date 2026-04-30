import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from scipy.stats import f_oneway

#Set up
os.makedirs("images", exist_ok=True)

#Load cleaned dataset 
df = pd.read_csv("data/cleaned_diabetic_data.csv")

print("Loaded cleaned dataset.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

#Separate predictors and target variable
X = df.drop(columns=["readmit_30"])
y = df["readmit_30"]

#Categorical columns from your final list
categorical_cols = ["age", "gender", "insulin", "diabetesMed", "admission_type_id"]

# One-hot encode categorical predictors
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\nEncoded feature shape:", X_encoded.shape)

#Correlation Analysis
corr_df = X_encoded.copy()
corr_df["readmit_30"] = y

corr_with_target = corr_df.corr(numeric_only=True)["readmit_30"].sort_values(ascending=False)

print("\nTop correlations with readmit_30:")
print(corr_with_target.head(15))
print("\nMost negative correlations with readmit_30:")
print(corr_with_target.tail(10))

plt.figure(figsize=(14, 10))
sns.heatmap(corr_df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression
log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred = log_model.predict(X_test_scaled)

print("\n================ LOGISTIC REGRESSION ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

log_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": log_model.coef_[0]
})
log_importance["Abs_Coefficient"] = log_importance["Coefficient"].abs()
log_importance = log_importance.sort_values(by="Abs_Coefficient", ascending=False)

print("\nTop Logistic Regression Predictors:")
print(log_importance[["Feature", "Coefficient"]].head(15))

top_log = log_importance.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_log, x="Coefficient", y="Feature")
plt.title("Top Logistic Regression Predictors")
plt.tight_layout()
plt.savefig("images/top_logistic_predictors.png")
plt.close()

# Remove age variables
log_importance = log_importance[
    ~log_importance["Feature"].str.contains("age")
]

# THEN select top features
top_log = log_importance.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_log, x="Coefficient", y="Feature")
plt.title("Top Logistic Regression Predictors")
plt.tight_layout()
plt.savefig("images/top_logistic_predictors.png")
plt.close()

# Model Evaluation Metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Calculate metrics
auc = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:,1])
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

# Create dataframe
performance_df = pd.DataFrame({
    "Metric": [
        "AUC", "Accuracy", "Recall", "Precision", "F1 Score", "Specificity",
        "True Negatives", "False Positives", "False Negatives", "True Positives"
    ],
    "Value": [
        auc,
        accuracy_score(y_test, y_pred),
        recall,
        precision,
        f1,
        specificity,
        tn, fp, fn, tp
    ]
})

# Round values for cleaner output
performance_df["Value"] = performance_df["Value"].round(3)

# Save CSV
performance_df.to_csv("images/model_evaluation_metrics.csv", index=False)

# Visualization
plt.figure(figsize=(8,5))
sns.barplot(data=performance_df.iloc[:6], x="Value", y="Metric")
plt.title("Logistic Regression Model Evaluation Metrics")
plt.xlim(0,1)
plt.tight_layout()
plt.savefig("images/model_evaluation_metrics.png")
plt.close()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

print("\n================ DECISION TREE ================")

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, dt_pred))
print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

# Feature importance
dt_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": dt_model.feature_importances_
})

dt_importance = dt_importance.sort_values(by="Importance", ascending=False)

print("\nTop Decision Tree Predictors:")
print(dt_importance.head(10))

#Remove Age Variables
dt_importance = dt_importance[
    ~dt_importance["Feature"].str.contains("age", case=False, na=False)
]

top_dt = dt_importance.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_dt, x="Importance", y="Feature")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.savefig("images/decision_tree_importance.png")
plt.close()

#Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

lin_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": lin_model.coef_
})
lin_importance["Abs_Coefficient"] = lin_importance["Coefficient"].abs()
lin_importance = lin_importance.sort_values(by="Abs_Coefficient", ascending=False)

print("\n================ LINEAR REGRESSION ================")
print("Top Linear Regression Coefficients:")
print(lin_importance[["Feature", "Coefficient"]].head(15))

top_lin = lin_importance.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_lin, x="Coefficient", y="Feature")
plt.title("Top Linear Regression Coefficients")
plt.tight_layout()
plt.savefig("images/top_linear_predictors.png")
plt.close()

lin_importance = lin_importance[
    ~lin_importance["Feature"].str.contains("age")
]

top_lin = lin_importance.head(10)

# Random Forest

from sklearn.ensemble import RandomForestClassifier
print("\n================ RANDOM FOREST ================")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Feature Importance
rf_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(by="Importance", ascending=False)

print("\nTop Random Forest Predictors:")
print(rf_importance.head(10))

# Remove Age Variables
rf_importance = rf_importance[
    ~rf_importance["Feature"].str.contains("age", case=False, na=False)
]

top_rf = rf_importance.head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_rf, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("images/random_forest_importance.png")
plt.close()

#ANOVA
print("\n================ ANOVA RESULTS ================")

anova_vars = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient",   
    "time_in_hospital",
    "num_medications",
    "num_procedures",      
    "number_diagnoses"
]

anova_results = []

for col in anova_vars:
    group0 = df[df["readmit_30"] == 0][col]
    group1 = df[df["readmit_30"] == 1][col]

    f_stat, p_val = f_oneway(group0, group1)
    anova_results.append({
        "Variable": col,
        "F_statistic": f_stat,
        "p_value": p_val
    })
    print(f"{col}: F = {f_stat:.4f}, p = {p_val:.10f}")

anova_df = pd.DataFrame(anova_results).sort_values(by="F_statistic", ascending=False)
anova_df.to_csv("images/anova_results.csv", index=False)

#Clustering
print("\n================ CLUSTERING ================")

cluster_features = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient",   
    "time_in_hospital",
    "num_medications",
    "num_procedures",     
    "number_diagnoses"
]

cluster_data = df[cluster_features].copy()

cluster_scaler = StandardScaler()
cluster_scaled = cluster_scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_scaled)

print("Cluster counts:")
print(df["cluster"].value_counts())

cluster_summary = df.groupby("cluster")[cluster_features + ["readmit_30"]].mean()
print("\nCluster summary:")
print(cluster_summary)

cluster_summary.to_csv("images/cluster_summary.csv")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="number_inpatient",
    y="num_medications",
    hue="cluster",
    palette="deep",
    alpha=0.6
)
plt.title("Patient Clusters: number_inpatient vs num_medications")
plt.tight_layout()
plt.savefig("images/clusters_scatter.png")
plt.close()

# Final selected predictors visual
final_vars = [
    "number_inpatient",
    "num_medications",
    "time_in_hospital",
    "number_diagnoses"
]

final_importance = rf_importance[rf_importance["Feature"].isin(final_vars)]

plt.figure(figsize=(9, 5))
sns.barplot(data=final_importance, x="Importance", y="Feature")
plt.title("Final Selected Predictors of 30-Day Readmission")
plt.xlabel("Random Forest Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig("images/final_selected_predictors.png")
plt.close()

# Readmission rate by number of prior inpatient visits
df["inpatient_group"] = pd.cut(
    df["number_inpatient"],
    bins=[-1, 0, 1, 2, 5, df["number_inpatient"].max()],
    labels=["0", "1", "2", "3-5", "6+"]
)

inpatient_rate = df.groupby("inpatient_group")["readmit_30"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=inpatient_rate, x="inpatient_group", y="readmit_30")
plt.title("30-Day Readmission Rate by Prior Inpatient Visits")
plt.xlabel("Prior Inpatient Visits")
plt.ylabel("Readmission Rate")
plt.tight_layout()
plt.savefig("images/readmission_by_inpatient_visits.png")
plt.close()

# Readmission rate by medication count
df["medication_group"] = pd.cut(
    df["num_medications"],
    bins=[0, 10, 20, 30, 40, df["num_medications"].max()],
    labels=["1-10", "11-20", "21-30", "31-40", "41+"]
)

med_rate = df.groupby("medication_group")["readmit_30"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=med_rate, x="medication_group", y="readmit_30")
plt.title("30-Day Readmission Rate by Number of Medications")
plt.xlabel("Number of Medications")
plt.ylabel("Readmission Rate")
plt.tight_layout()
plt.savefig("images/readmission_by_medications.png")
plt.close()

# Model accuracy comparison
model_scores = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=model_scores, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("images/model_accuracy_comparison.png")
plt.close()

# Model accuracy comparison
model_scores = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=model_scores, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("images/model_accuracy_comparison.png")
plt.close()

#Summary
print("\n================ FINAL SUMMARY ================")
print("Based on correlation, regression, ANOVA, and clustering,")
print("the strongest predictors are expected to be centered around:")
print("- number_inpatient")
print("- num_medications")
print("- time_in_hospital")
print("- number_diagnoses")
print("\nAnalysis complete. Check the images folder for visuals.")