from flask import Flask, render_template, request
import jsonify
import requests
import numpy as np
import keras
import tensorflow as tf
import io
import json
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
from keras_preprocessing.text import tokenizer_from_json

nltk.download('stopwords')

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def text_preprocess(text):
    text=text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


app = Flask(__name__)
classifier = keras.models.load_model('model/')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['Email']
        email= pd.Series([email])
        email=email.apply(text_preprocess)
        email_features = np.array(tokenizer.texts_to_sequences(email))
        email_features=pad_sequences(email_features, maxlen=72)
        output=classifier.predict(email_features)
        if output>0.5:
            return render_template('index.html',prediction_texts="This input Email is spam")
        else:
            return render_template('index.html',prediction_text="This input Email is not spam")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

