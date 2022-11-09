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



class LSTM_NET_2(nn.Module):
    def __init__(self, n_embed, n_hidden, n_vocab):
        super(LSTM_NET_2, self).__init__()
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm1 = nn.LSTM(n_embed, n_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.l1 = nn.Linear(n_hidden, n_vocab)

    def forward(self, target, hidden1, hidden2):
        e = self.embedding(target)
        y1, hidden1 = self.lstm1(e) if hidden1 is None else self.lstm1(e, hidden1)
        y2, hidden2 = self.lstm2(y1) if hidden2 is None else self.lstm1(y1, hidden2)
        v = self.l1(y2)
        return v, hidden1, hidden2

class LSTM_NET_3(nn.Module):
    def __init__(self, n_embed, n_hidden, n_vocab):
        super(LSTM_NET_3, self).__init__()
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(n_embed, n_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.l1 = nn.Linear(n_hidden, n_vocab)
        self.l1.weight = self.embedding.weight

    def forward(self, target, hidden1, hidden2):
        e = self.embedding(target)
        e = self.drop1(e)
        y1, hidden1 = self.lstm1(e) if hidden1 is None else self.lstm1(e, hidden1)
        y1 = self.drop2(y1)
        y2, hidden2 = self.lstm2(y1) if hidden2 is None else self.lstm1(y1, hidden2)
        y2 = self.drop3(y2)
        v = self.l1(y2)
        return v, hidden1, hidden2