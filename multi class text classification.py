#exp-6
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
maxlen=500
train_data=pad_sequences(train_data,maxlen=maxlen)
test_data =pad_sequences(test_data,maxlen=maxlen)
num_classes=len(set(train_labels))
train_labels=to_categorical(train_labels,num_classes)
test_labels=to_categorical(test_labels,num_classes)

model=Sequential()
model.add(Embedding(10000,128,input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(num_classes,activation='Softmax'))
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
history=model.fit(train_data,train_labels,epochs=10,batch_size=128,validation_split=0.2)
test_loss,test_acc=model.evaluate(test_data,test_labels)
print("Test accuracy:",test_acc)
import matplotlib.pyplot as plt
train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
plt.figure(figsize=(10,5))
plt.plot(train_loss,label="Training Loss")
plt.plot(val_loss,label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.figure(figsize=(10,5))
plt.plot(train_acc,label="Training Accuracy")
plt.plot(val_acc,label="Validation Loss")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
predicted_probs=model.predict(test_data)
predicted_labels=np.argmax(predicted_probs,axis=1)
test_labels_cat=np.argmax(test_labels,axis=1)
conf_matrix=confusion_matrix(test_labels_cat,predicted_labels)
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="Blues")
plt.title("Confusion_matrix")
plt.xlabel('Predicted_label')
plt.ylabel("True Label")
plt.show()
