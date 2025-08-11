import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from jinja2.environment import load_extensions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
from tensorflow.keras.models import load_model


# Read the file
data = pd.read_csv("Churn_Modelling.csv")

# remove unwanted columns
data = data.drop(["CustomerId", "RowNumber", "Surname"], axis=1)

# make the GENDER values to binary 0 or 1
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# Onehot encode Geo column
onehotencoder_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoder = onehotencoder_geo.fit_transform(data[['Geography']]).toarray()

# create a new data frame
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

# drop the GEOGRAPHY column and add new columns from geo_encoded_df
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

#divide the dataset into dependent and independent features
x = data.drop('Exited', axis=1)
y = data['Exited']

# split the data in training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# save the encoders to picke file
with open("label_encoder_gender.pkl", 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open("onehotencoder_geo.pkl", 'wb') as file:
    pickle.dump(onehotencoder_geo, file)



# Build the ANN Model (fixed output activation)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # binary probability output
])

# Optimizer and loss (matched to sigmoid)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

log_dir = "logs/fit/"
tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train on SCALED data
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping, tensorflow_callback],
    verbose=1
)

with open("scaler.pkl", 'wb') as file:
    pickle.dump(scaler, file)

model.save('model.h5')

load_model('model.h5')
