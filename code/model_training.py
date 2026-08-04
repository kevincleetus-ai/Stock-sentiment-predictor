import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("data/AAPL_features.csv")

# price and sentiment features
features = ["Open", "High", "Low", "Close", "Volume", "avg_sentiment", "num_headlines", "price_change_pct", "ma_7", "ma_30", "volatility", "volume_change_pct", "rsi"]
X = df[features]
y = df["target"]

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.2%}")

xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f"XGBoost Accuracy: {xgb_accuracy:.2%}")

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))

# xgboost performed better so saving that one
joblib.dump(xgb_model, "models/xgb_model.pkl")
print("Model saved!")