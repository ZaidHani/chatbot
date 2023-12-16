from chatbot import *
import random
import streamlit as st

sentance = st.text_input("You: ")
sentance = tokenize(sentance)
x = bag_of_words(sentance, all_words)
x = x.reshape(1, x.shape[0])
x = torch.from_numpy(x)

output = model(x)
_, predicted = torch.max(output, dim=1)
tag = tags[predicted.item()]

for intent in intetns['intents']:
    if tag==intent['tag']:
        st.markdown(f"BOT: {random.choice(intent['responses'])}")