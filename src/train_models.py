import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ======================================================
# Load data
# ======================================================

df = pd.read_csv("data/processed/PING_PONG_FEATURES.csv")

drop_cols = ["Unnamed: 0", "index", "id", "timestamp", "trial", "player"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ======================================================
# Target Encoding
# ======================================================

target_col = "action"
encoder = LabelEncoder()
y = encoder.fit_transform(df[target_col])
class_names = encoder.classes_

X = df.drop(columns=[target_col])

# ======================================================
# Train / Test Split + Scaling
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# Models
# ======================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    ),
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=2000)
}

# ======================================================
# Cross-Validation
# ======================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# ======================================================
# Train & Evaluate
# ======================================================

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n" + "=" * 50)
    print(f"{name} — Test Accuracy: {acc:.4f}")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=class_names))

# ======================================================
# Model Comparison Plot  
# ======================================================

plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Model Comparison on Ping-Pong Stroke Dataset")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# Confusion Matrix 
# ======================================================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix – {best_model_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# Feature Importance 
# ======================================================

rf = models["Random Forest"]
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.xlabel("Importance Score")
plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
