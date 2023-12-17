# this file is runned once and only once, no need to run it more than one time and waste resources


# importing libraries
import nltk 
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import string
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf


# reading the data
df =open('intents.json', encoding='utf-8').read()
data=json.loads(df)

# NLP mumbo jumbo
words = []
classes = []
data_x = [] 
data_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_x.append(pattern)
        data_y.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

# more NLP mumbo jumbo
out_empty = [0] * len(classes)
training = []

for idx, doc in enumerate(data_x):
    bow = [] 
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        if word in text:
            bow.append(1)
        else:
            bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    
    training.append((bow, output_row))

random.shuffle(training)

train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

# building the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))  # Assuming this is a multi-class classification task

# Use the legacy optimizer
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
# print(model.summary())

# fitting the data into the model
model.fit(x=train_x, y=train_y,epochs=150,verbose=1)

model.save('model.model')