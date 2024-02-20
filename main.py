import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk

model = joblib.load('sentiment_model.pkl')
stopwords = list(STOP_WORDS)

app = Flask(__name__)

from tokenizer import CustomTokenizer

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]

    predictions = model.predict(new_review)[0]
    if predictions == 'no':
        return render_template('./index.html', prediction_text='Negative')
    else:
        return render_template('./index.html', prediction_text='Positive')

if __name__ == "__main__":
    app.run(debug=True)