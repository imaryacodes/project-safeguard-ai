# SafeGuard AI — Spam Email Detector

An AI-powered spam email detection system built using Natural Language Processing and Machine Learning techniques. The system classifies emails as **Spam** or **Ham (Legitimate)** using a Naive Bayes Classifier.


## Objective

To develop a system that automatically detects and classifies emails as spam or legitimate using NLP and Machine Learning — helping users filter harmful emails, reduce phishing exposure, and improve digital safety.


## How It Works

```
Email Text Input → CountVectorizer → TF-IDF Transformation → Naive Bayes Classifier → Spam / Ham Output
```
1. Email text is loaded, cleaned, and labelled (ham = 0, spam = 1)
2. Data is split into **80% training / 20% testing** using stratified sampling
3. A scikit-learn Pipeline processes and classifies the text:
   - **CountVectorizer** — converts text into word frequency vectors (max 3,000 features)
   - **TF-IDF Transformer** — weights words by importance across the dataset
   - **Multinomial Naive Bayes** — performs final spam/ham classification


## AI / ML Techniques Used

- **Natural Language Processing (NLP)** — processes raw email text
- **Bag-of-Words Model (CountVectorizer)** — converts text to numerical vectors
- **TF-IDF Transformation** — assigns importance weights to words
- **Multinomial Naive Bayes Classifier** — probabilistic text classification
- **Laplace Smoothing (alpha = 1.0)** — handles unseen vocabulary
- **5-Fold Stratified Cross-Validation** — validates model and prevents overfitting
- **Evaluation Metrics** — accuracy, precision, recall, F1-score, confusion matrix
- **Matplotlib & Seaborn** — visualising confusion matrix and cross-validation results

---

## Libraries & Tools

| Library | Purpose |
|---|---|
| `scikit-learn` | ML pipeline, Naive Bayes, vectorizer, metrics |
| `pandas` | Data loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting graphs |
| `seaborn` | Confusion matrix visualisation |

---

## Dataset

**SMS Spam Collection Dataset** — sourced from Kaggle

| Detail | Info |
|---|---|
| Total Messages | 5,574 |
| Ham (Legitimate) | 4,827 |
| Spam | 747 |
| Format | CSV (label + message text) |

🔗 [Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## Key Features

- Automatically detects spam emails without manual screening
- Protects users from phishing attacks and fraudulent content
- Trained on a real-world dataset for genuine spam patterns
- Highlights keywords that triggered spam classification
- Cross-validated to ensure the model generalises well

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/imaryacodes/project-safeguard-ai.git
   cd project-safeguard-ai
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```

3. Run the project:
   ```bash
   python app.py
   ```

---

## Author

**Arya Prakash**
B.Tech Computer Science & Engineering — Bennett University

---

## 📌 Status

✅ **Completed** — Fully working spam detection system
