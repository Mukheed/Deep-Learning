#exp-8
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
# Parameters
max_features = 10000 # Number of words to consider as features
maxlen = 500 # Cuts off texts after this many words (among the max_features most common words)
batch_size = 32
embedding_dim = 50
epochs = 5
# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
# Define the model
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(SimpleRNN(32)) # You can adjust the number of units as needed
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
