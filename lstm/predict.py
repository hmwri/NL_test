import pickle
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
model.load_state_dict(torch.load(model_path))

target = torch.tensor([word_to_id["私"]])
model.eval()
softmax = torch.nn.Softmax()
for i in range(10):
    y = model.forward(target, None, None)
    y = softmax(y)
    print(y)