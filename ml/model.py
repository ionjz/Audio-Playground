import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .features import extract_features_for_training, extract_features

def build_training_dataframe(pairs):
  rows = [extract_features_for_training(p) for p in pairs]
  return pd.DataFrame(rows)

def train_classifier(df: pd.DataFrame):
  X = df.drop(columns=['IsExtremist'])
  y = df['IsExtremist']
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
  clf = RandomForestClassifier(n_estimators=200, random_state=42)
  clf.fit(X_train, y_train)
  return clf, scaler, (X_test, y_test)

def evaluate_classifier(clf, test_tuple):
  X_test, y_test = test_tuple
  from sklearn.metrics import accuracy_score, classification_report
  y_pred = clf.predict(X_test)
  return {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'report': classification_report(y_test, y_pred, zero_division=0)
  }

def predict_extremism(clf, scaler, sentence: str):
  row = extract_features(sentence)
  df = pd.DataFrame([row])
  X_scaled = scaler.transform(df)
  pred = clf.predict(X_scaled)
  return int(pred[0])

def predict_proba_extremism(clf, scaler, sentence: str) -> float:
  row = extract_features(sentence)
  df = pd.DataFrame([row])
  X_scaled = scaler.transform(df)
  if hasattr(clf, 'predict_proba'):
    proba = clf.predict_proba(X_scaled)
    # Probability of class 1
    return float(proba[0][1])
  # Fallback: 1.0 if predicted class==1 else 0.0
  return float(clf.predict(X_scaled)[0])

def save_model(artifact_path: str, clf, scaler):
  os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
  with open(artifact_path, 'wb') as f:
    pickle.dump({'clf': clf, 'scaler': scaler}, f)

def load_model(artifact_path: str):
  with open(artifact_path, 'rb') as f:
    obj = pickle.load(f)
  return obj.get('clf'), obj.get('scaler')


