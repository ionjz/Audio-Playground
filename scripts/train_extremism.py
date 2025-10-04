import argparse
import os
import sys

# Allow running as a script: `python scripts/train_extremism.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

from ml import synthetic_sentences
from ml.model import (
  build_training_dataframe,
  train_classifier,
  evaluate_classifier,
  predict_extremism,
  predict_proba_extremism,
  save_model,
)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict', type=str, default=None, help='Predict label for a sentence')
  parser.add_argument('--save', type=str, default='models/extremism_rf.pkl', help='Path to save trained model')
  args = parser.parse_args()

  df = build_training_dataframe(synthetic_sentences)
  clf, scaler, test_tuple = train_classifier(df)
  metrics = evaluate_classifier(clf, test_tuple)
  print(f"Accuracy: {metrics['accuracy']:.4f}")
  print(metrics['report'])
  save_model(args.save, clf, scaler)
  print(f"Saved model to {args.save}")

  if args.predict:
    prob = predict_proba_extremism(clf, scaler, args.predict)
    label = int(prob >= 0.5)
    print('Predicted IsExtremist:', label)
    print(f'Probability: {prob:.3f}')

if __name__ == '__main__':
  main()


