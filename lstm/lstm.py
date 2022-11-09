import os
import sys
sys.path.append('..')
import numpy as np
import torch

from read_file import *
from preprocess import *
from dataset import LSTM_Dataset, LSTM_DataLoader
import pickle
import net
import torch.nn as nn
from lstm_trainer import *

def load_corpus(reload = False, short=False):
    filename = "short_corpus" if short else "corpus"
    if reload:
        if short:
            texts = read_corpus_file("../text",n=1000)
        else:
            texts = read_corpus_file("../text")
        texts = preprocess(texts)
        print(f"read : {texts}")
        corpus, word_to_id, id_to_word = make_corpus(texts)
        print(corpus)


        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump({
                "corpus":corpus,
                "word_to_id":word_to_id,
                "id_to_word":id_to_word,
            }, f)


    with open(f'{filename}.pkl', 'rb') as f:
        data = pickle.load(f)
        corpus = data["corpus"]
        word_to_id = data["word_to_id"]
        id_to_word = data["id_to_word"]

    return corpus, word_to_id, id_to_word



corpus, word_to_id, id_to_word = load_corpus(reload=False,short=False)
print(len(word_to_id))

time_step = 20
ds = LSTM_Dataset(corpus, time_step,train=True)
ds_v = LSTM_Dataset(corpus, time_step,train=False)
test_ds = LSTM_Dataset(list(range(20)), 3 ,train=True)
test_dl = LSTM_DataLoader(test_ds,batch_size=3)
for x, i in test_dl:
    print(x)
print(len(ds))
N = 100
dl = LSTM_DataLoader(ds, batch_size=N)
dl_eval = LSTM_DataLoader(ds_v, batch_size=N)
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
epoch_num = 15

Net = net.LSTM_NET(n_embed,H,V)
# for x, y in tqdm(dl):
#     break
criterion = nn.CrossEntropyLoss()
#
# h_0 = torch.zeros(1, N, H)
# c_0 = torch.zeros(1, N, H)
# output, h_out, c_out = Net(x,h_0,c_0)

# loss = criterion(output.view(-1, V), y.view(-1))
# print(loss, h_out, c_out)
# loss.backward()

# img = make_dot(loss, params=dict(Net.named_parameters()))
# img.format = "png"
# img.render("NeuralNet")

optim = torch.optim.Adam(Net.parameters())
trainer = LstmTrainer(Net,criterion,optim,name="lstm_trainer_2_short")

trainer.fit(epoch_num,dl,dl_eval)


