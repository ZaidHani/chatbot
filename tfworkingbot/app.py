from nlp_bot import pred_class, get_response, words, data, classes
import streamlit as st
import tensorflow as tf

# making a header and a title for the website
st.title('Chatbot')
st.markdown('This chatbot was made by using NLTK and Tensorflow, this is only a test chatbot.')

#

message = st.text_input("You: ")

intents = pred_class(message, words, classes)
result = get_response(intents, data)
st.markdown(f'Bot: {result}')
