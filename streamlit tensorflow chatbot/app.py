from src.nltk_utils import bag_of_words, tokenize, all_words, tags
from src.model import NeuralNet
import torch
import random
import streamlit as st
import json

# loading the data
json_file = '../intents.json'
intents = json.loads(open(json_file).read())

# loading the model
FILE = 'data1.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)

# streamlit stuff
st.title('Chatbot')
st.markdown('this is the data science chatbot, feel free to ask it anything related to data sicence, this chatbot was build using Tensorflow and NLTK')

# giving the bot input and doing ML by getting the bag of words
sentance = st.text_input("You: ")
sentance = tokenize(sentance)
x = bag_of_words(sentance, all_words)
x = x.reshape(1, x.shape[0])
x = torch.from_numpy(x)

# taking the input 'x', putting it in the model and getting the output
output = model(x)
_, predicted = torch.max(output, dim=1)
tag = tags[predicted.item()]

# displaying the ML result
for intent in intents['intents']:
    if tag==intent['tag']:
        st.markdown(f"BOT: {random.choice(intent['responses'])}")