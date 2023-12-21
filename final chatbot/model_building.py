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
# dropout helps to avoid overfitting
# avtivation make the nuearal network learn the cmoplex patterns and realtionship
# without the activation function the NN will act like a linear model 
# the optimizer adam don't need to put the wieght

model = Sequential([
    # layer 1
    (Dense(128, input_shape=(len(train_x[0]),), activation="relu")),
    (Dropout(0.5)),
    # layer 2
    (Dense(64, activation="relu")),
    (Dropout(0.5)),
    # layer 3
    (Dense(len(train_y[0]), activation="softmax"))
])

# Use the legacy optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
# print(model.summary())

# fitting the data into the model
model.fit(x=train_x, y=train_y,epochs=500,verbose=1)

model.save('model.model')