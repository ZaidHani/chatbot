from nlp_bot import pred_class, get_response, words, data, classes
import streamlit as st
import tensorflow as tf

# making a header and a title for the website

#

message = input("You: ")

intents = pred_class(message, words, classes)
result = get_response(intents, data)
print(f'Bot: {result}')
