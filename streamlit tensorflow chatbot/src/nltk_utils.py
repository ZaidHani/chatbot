from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import numpy as np


lem = WordNetLemmatizer()
stm = PorterStemmer()

def tokenize(sentance):
    return nltk.word_tokenize(sentance)

def lemm(word):
    return lem.lemmatize(word.lower())

def stem(word):
    return stm.stem(word.lower())

# this is probably the most important function, try to understand what it does
def bag_of_words(tokenized_sentance, all_words):
    tokenized_sentance = [lemm(word) for word in tokenized_sentance]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index_of_the_word, word in enumerate(all_words):
        if word in tokenized_sentance:
            bag[index_of_the_word] = 1.0
    return bag