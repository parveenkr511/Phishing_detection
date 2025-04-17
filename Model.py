import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Feature Extraction Function
# -------------------------------
def extract_features(URL):
    return {
        'url_length': len(URL),
        'num_dots': URL.count('.'),
        'num_hyphens': URL.count('-'),
        'num_at': URL.count('@'),
        'has_https': int(URL.startswith("https")),
        'has_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', URL))),
        'num_subdirs': URL.count('/'),
        'num_parameters': URL.count('='),
        'num_percent': URL.count('%'),
        'num_www': URL.count('www'),
        'num_digits': sum(c.isdigit() for c in URL),
        'num_letters': sum(c.isalpha() for c in URL)
    }

# -------------------------------
# 2. Load Dataset
# -------------------------------
# CSV should have 'url' and 'label' columns
data = pd.read_csv("processed_dataset.csv")

# -------------------------------
# 3. Feature Engineering
# -------------------------------
features = data['URL'].apply(extract_features)
X = pd.DataFrame(features.tolist())
y = data['label']

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Base Learners
# -------------------------------
from xgboost import XGBClassifier

xgb = XGBClassifier(eval_metric='logloss', random_state=42)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb),
    ('svm', SVC(probability=True, kernel='rbf', C=1.0)),
    ('lr', LogisticRegression(max_iter=1000))
]

# -------------------------------
# 6. Stacking Ensemble Model
# -------------------------------
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5
)

# -------------------------------
# 7. Train the Model
# -------------------------------
print("Training started...")
stacked_model.fit(X_train, y_train)
print("Training complete.")

# -------------------------------
# 8. Evaluate the Model
# -------------------------------
y_pred = stacked_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 9. Predict New URL
# -------------------------------
def predict_url(URL):
    features = extract_features(URL)
    df = pd.DataFrame([features])
    prediction = stacked_model.predict(df)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

# Example prediction
print("Executing...")
print(predict_url("http://verify-paypal.com/secure-login"))

