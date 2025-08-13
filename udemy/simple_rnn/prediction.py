import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, LSTM, Dense
import os
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {val:key for key, val in word_index.items()}

#load the model
model = load_model('imdb.h5')
print(model.summary())

#decode the reviews
def decode_review(encoded_review):
    res = ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])
    print('res = ', res)
    return res

# preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    print('words = ', words)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    print('encoded = ',encoded_review)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

ex_review = "This movie was fantastic and The acting was great and the plot was thrilling."
sentiment, score = predict_sentiment(ex_review)
print('sentiment = ', sentiment)
print('score = ', score)

