import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
# Load the dataset
df = pd.read_csv("data/AAPL_features.csv")
# Select features and target
features = ["Open", "High", "Low", "Close", "Volume", "avg_sentiment", "num_headlines"]
X = df[features]
y = df["target"]
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")
# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.2%}")

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f"XGBoost Accuracy: {xgb_accuracy:.2%}")

# Detailed report for best model
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))