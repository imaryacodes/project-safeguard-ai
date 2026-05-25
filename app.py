import os
import pickle
import webbrowser
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# Load trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    spam_idx = list(model.classes_).index('spam')
    ham_idx  = list(model.classes_).index('ham')

    spam_prob = round(proba[spam_idx] * 100, 1)
    ham_prob  = round(proba[ham_idx]  * 100, 1)
    confidence = spam_prob if prediction == 'spam' else ham_prob

    return jsonify({
        'label':      prediction,
        'confidence': confidence,
        'spam_prob':  spam_prob,
        'ham_prob':   ham_prob
    })


def open_browser():
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000)
