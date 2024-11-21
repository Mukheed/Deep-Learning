#exp5
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
vocab_size=10000
max_len=200
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=vocab_size)
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
random_state=42)
model_gru = Sequential()
model_gru.add(Embedding(vocab_size, 128, input_length=max_len))
model_gru.add(GRU(64))
model_gru.add(Dense(2, activation='softmax'))
model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_gru.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=64)
gru_loss, gru_accuracy = model_gru.evaluate(x_test, y_test)
print(f"GRU Model - Test Loss: {gru_loss:.4f}, Test Accuracy: {gru_accuracy:.4f}")
model_lstm = Sequential()
model_lstm.add(Embedding(vocab_size, 128, input_length=max_len))
model_lstm.add(LSTM(64))
model_lstm.add(Dense(2, activation='softmax'))
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=64)
lstm_loss, lstm_accuracy = model_lstm.evaluate(x_test, y_test)
print(f"LSTM Model - Test Loss: {lstm_loss:.4f}, Test Accuracy: {lstm_accuracy:.4f}")