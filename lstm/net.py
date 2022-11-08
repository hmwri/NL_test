import torch
import torch.nn as nn
class LSTM_NET(nn.Module):

    def __init__(self, n_embed, n_hidden, n_vocab):
        super(LSTM_NET, self).__init__()
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, batch_first=True)
        self.l1 = nn.Linear(n_hidden, n_vocab)

    def forward(self, target, h_0=0, c_0=0):
        e = self.embedding(target)
        y, (h_out, c_out) = self.lstm(e,(h_0,c_0))
        v = self.l1(y)
        return v, h_out, c_out




