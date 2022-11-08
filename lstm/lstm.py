import os

import torch

from read_file import *
from preprocess import *
from dataset import LSTM_Dataset, LSTM_DataLoader
from torch.utils.data import DataLoader
from torchviz import make_dot
import pickle
import net
import torch.nn as nn
from IPython.display import display

reload_copus = False
if reload_copus:
    texts = read_corpus_file("../text")
    texts = preprocess(texts)
    print(f"read : {texts}")
    corpus, word_to_id, id_to_word = make_corpus(texts)
    print(corpus)


    with open('corpus.pkl', 'wb') as f:
        pickle.dump({
            "corpus":corpus,
            "word_to_id":word_to_id,
            "id_to_word":id_to_word,
        }, f)


with open('corpus.pkl', 'rb') as f:
    data = pickle.load(f)
    corpus = data["corpus"]
    word_to_id = data["word_to_id"]
    id_to_word = data["id_to_word"]


print(corpus)

time_step = 10
ds = LSTM_Dataset(corpus, time_step)

print(len(ds))
N = 100
dl = LSTM_DataLoader(ds, batch_size=N)
# i = 0
# for x, y in tqdm(dl):
#     if i == dl.length - 1:
#         for xx,yy in zip(x, y):
#             print(to_word(xx, id_to_word))
#             print(to_word(yy, id_to_word))
#     i+=1

n_embed = 10
H = 10
V = len(word_to_id)
epoch_num = 10

Net = net.LSTM_NET(n_embed,H,V)
for x, y in tqdm(dl):
    break
criterion = nn.CrossEntropyLoss()

h_0 = torch.zeros(1, N, H)
c_0 = torch.zeros(1, N, H)
output, h_out, c_out = Net(x,h_0,c_0)

loss = criterion(output.view(-1, V), y.view(-1))
print(loss, h_out, c_out)
loss.backward()

# img = make_dot(loss, params=dict(Net.named_parameters()))
# img.format = "png"
# img.render("NeuralNet")

device = "cpu"
if torch.backends.mps.is_available() :
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
device = torch.device(device)
Net = Net.to(device)
optim = torch.optim.Adam(Net.parameters())

for epoch in range(epoch_num):
    h_0 = torch.zeros(1, N, H,dtype=torch.float).to(device)
    c_0 = torch.zeros(1, N, H, dtype=torch.float).to(device)
    total_loss = 0
    Net.train()
    for x,y in tqdm(dl):
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        output, h_out, c_out = Net(x, h_0, c_0)
        loss = criterion(output.view(-1,V), y.view(-1))
        loss.backward()
        optim.step()
        total_loss += loss
    print(epoch,total_loss.item())


