import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], header=0)
df.columns = ['label', 'text']
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ── Load saved model & vectorizer ─────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

X_test_vec = vectorizer.transform(X_test)
y_pred     = model.predict(X_test_vec)

# ── Confusion Matrix ──────────────────────────────────────────
labels = ['ham', 'spam']
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\n" + "="*50)
print("           CONFUSION MATRIX")
print("="*50)
print(f"\n              Predicted HAM   Predicted SPAM")
print(f"  Actual HAM      {cm[0][0]:<10}     {cm[0][1]:<10}")
print(f"  Actual SPAM     {cm[1][0]:<10}     {cm[1][1]:<10}")

# ── Evaluation Metrics ────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='spam')
rec  = recall_score(y_test, y_pred, pos_label='spam')
f1   = f1_score(y_test, y_pred, pos_label='spam')

print("\n" + "="*50)
print("           EVALUATION METRICS")
print("="*50)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")

print("\n" + "="*50)
print("       FULL CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=labels))
