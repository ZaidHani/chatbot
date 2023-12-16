from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

intents = json.loads(open('intents.json').read())

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
clean_words = [word for word in all_words if word not in ignore_words]

# lowering words
lower_words = [word.lower() for word in clean_words]

# sorting words
sortd_words = sorted(set(lower_words))

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


# som pytroch action

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

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation, no selfmax
        return out
    
# more hyperparameters
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
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

# saving the model
def save():
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)