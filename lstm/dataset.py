from torch.utils.data import Dataset
import torch


class LSTM_Dataset(Dataset):
    def __init__(self, corpus, time_step,train=True,split_rate=0.9):
        if train:
            corpus = corpus[:int(len(corpus) * split_rate)]
        else:
            corpus = corpus[int(len(corpus) * split_rate):]
        self.x = torch.tensor(corpus[:-1],dtype=torch.long)
        self.y = torch.tensor(corpus[1:],dtype=torch.long)
        self.time_step = time_step

    def __len__(self):
        return len(self.x) // self.time_step

    def __getitem__(self, index):
        x = self.x[index*self.time_step:(index+1)*self.time_step]
        y = self.y[index*self.time_step:(index+1)*self.time_step]
        return x, y



class LSTM_DataLoader(object):
    def __init__(self,dataset,batch_size=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.length = len(dataset) // batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.index == self.length:
            self.index = 0
            raise StopIteration()
        xset = torch.zeros(self.dataset.time_step, self.batch_size, dtype=torch.long)
        yset = torch.zeros(self.dataset.time_step, self.batch_size,dtype=torch.long)
        for i in range(self.batch_size):
            x,y = self.dataset[self.index+i*self.length]
            for t, (xt, yt) in enumerate(zip(x,y)):
                xset[t][i] = xt
                yset[t][i] = yt
        self.index += 1
        return xset,yset
