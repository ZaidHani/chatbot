import json
import numpy as np
from nltk_utils import  tokenize, stem, lemm, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import NeuralNet

intents = json.loads(open('../intents.json').read())

all_words = []
tags =[]
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        
        # notice here that we have uesd extend, becaues append will add each tokenized pattern as an indvidual list and we don't want that,
        # we want a 1D array not a multidimensional array
        # append -> ['google'] + ['hello'] = [['google'], ['hello']]
        # extedn -> ['google'] + ['hello'] = ['google', 'hello']
        
        all_words.extend(w)
        xy.append((w, tag))

# removing punctioation marks
ignore_words = [',', '.', '!', '?']
all_words = [word for word in all_words if word not in ignore_words]

# sorting words
all_words = sorted(set(all_words))

# we sorted the tags so we can have an easier job when we do the next step
lower_tags = [tag.lower() for tag in tags]
sorted_tags = sorted(lower_tags)

x_train = []
y_train =[]

for (pattern_sentance, tag) in xy:
    bag = bag_of_words(pattern_sentance, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # this is basiclly one_hot_encoding the tags or as it called in pytorch CrossEntropyLoss

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.astype('longlong')

#-------------------------------------------------------------------------------------------------------------------------------
# and now for some pytorch action
#-------------------------------------------------------------------------------------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
       
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
# hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])

learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          # num_workers is how many cores the cpu will use, put the num of workers to 0 if you wanted it to work
                          num_workers=0)

# training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        # forward
        output = model(words)
        loss = criterion(output, labels)

        # backward and optimizer steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')
print(f'final loss, loss = {loss.item():.4f}')
print('the model was successfully trained')

data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

FILE = "../data1.pth"
torch.save(data, FILE)
print(f'the file was saved in {FILE}')