# importing libraries
import nltk 
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import string
import random
import tensorflow as tf


# reading the data
df =open('intents.json', encoding='utf-8').read()
data=json.loads(df)

# NLP mumbo jumbo
# tokenizing, lemmatizing, and classifying text
words = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

# some functions to build the bot

model = tf.keras.models.load_model('model.model')

#Preprocessing the Input
def clean_text(text):
    tokens =nltk.word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word ==w:
                bow[idx] =1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list)==0:
        result="Soory! I don't unerstand"
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"]==tag:
                result=random.choice(i["responses"])
                break
    return result