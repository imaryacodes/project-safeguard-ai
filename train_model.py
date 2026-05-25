import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], header=0)
df.columns = ['label', 'text']
df.dropna(inplace=True)

print(f"Dataset loaded: {len(df)} rows  |  spam: {(df.label=='spam').sum()}  ham: {(df.label=='ham').sum()}")

# ── Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ── Vectorize ─────────────────────────────────────────────────
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Train ─────────────────────────────────────────────────────
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ── Save ──────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("model.pkl and vectorizer.pkl saved successfully.")
print(f"Test accuracy: {model.score(X_test_vec, y_test)*100:.2f}%")
