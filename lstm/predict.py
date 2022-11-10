import pickle
import preprocess
import numpy
import numpy as np


import torch
import net
with open(f'corpus.pkl', 'rb') as f:
    data = pickle.load(f)
    corpus = data["corpus"]
    word_to_id = data["word_to_id"]
    id_to_word = data["id_to_word"]



n_embed = 100
H = 100
V = len(word_to_id)
epoch_num = 50

model = net.LSTM_NET_2(n_embed,H,V)

model_path = 'model.pth'
model.load_state_dict(torch.load(model_path,map_location="cpu"))
text=input()
text = preprocess.preprocess(text,predict=True)

print(text)

target = [word_to_id.get(w,word_to_id["<UNK>"]) for w in text]
model.eval()
softmax = torch.nn.Softmax()
result = ""
with torch.no_grad():
    for i in range(50):
        t = torch.tensor(target)
        y,_,_ = model.forward(t, None, None)
        p = softmax(y[-1]).numpy()
        print([id_to_word[w] for w in numpy.argsort(p)][::-1][:5])
        w = np.random.choice(list(id_to_word),p=p)
        if w != word_to_id["<UNK>"]:
            print(id_to_word[w])
            target.append(w)
            result += id_to_word[w]

print(result)
