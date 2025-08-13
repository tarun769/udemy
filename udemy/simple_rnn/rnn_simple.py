import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, LSTM, Dense
import os
from tensorflow.keras.callbacks import EarlyStopping

# # Get the physical GPUs
# gpus = tf.config.list_physical_devices('GPU')
#
# # Directly set visible device to the first GPU (index 0)
# tf.config.set_visible_devices(gpus[0], 'GPU')

# # Optionally enable memory growth (recommended)
# tf.config.experimental.set_memory_growth(gpus[0], True)

# # Now TensorFlow will use only that single GPU
# print("Using GPU:", gpus[0])

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
max_features = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index() # prints {words:vectors} 'sunnybrook': 88134, 'memorializing': 88135, 'backlighting': 37506,

# make len of each sent equal by prefixing
max_len = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# build model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#define early stopping
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

model.save('imdb.h5')