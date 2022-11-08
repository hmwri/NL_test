import torch
from common.trainer import Trainer
from tqdm import tqdm
import numpy as np
from torchinfo import summary
import matplotlib.pyplot as plt
from IPython import display

class LstmTrainer(Trainer):
    def __init__(self,model, criterion,optimizer,device=None,name="test"):
        self.name = name
        super().__init__(model, criterion,optimizer,device)
        summary(self.model)
        self.H = self.model.n_hidden
        self.V = self.model.n_vocab
        self.info = {"name": self.name, "hidden":self.H, "vocab":self.V}

    def fit(self,epoch_num,data_loader,data_loader_eval):
        torch.save(self.model, f'{self.name}.pth')
        self.N = data_loader.batch_size
        interval = 100
        self.info["epoch"] = epoch_num
        self.info["batch_size"] = self.N
        self.info["time_step"] = data_loader.dataset.time_step
        self.info["interval"] = interval
        self.info["history_train"] = []
        self.info["history_eval"] = []
        def eval_proxity():
            loss_sum = 0
            h_0 = torch.zeros(1, self.N, self.H, dtype=torch.float).to(self.device)
            c_0 = torch.zeros(1, self.N, self.H, dtype=torch.float).to(self.device)
            with torch.no_grad():
                self.model.eval()
                for x, t in data_loader_eval:
                    x = x.to(self.device)
                    t = t.to(self.device)
                    output, h_0, c_0 = self.model(x, h_0, c_0)

                    loss = self.criterion(output.view(-1, self.V), t.view(-1))
                    loss_sum += loss.item()
                ppl = np.exp(loss_sum / len(data_loader_eval))
                return ppl

        for epoch in range(epoch_num):
            h_0 = torch.zeros(1, self.N, self.H, dtype=torch.float).to(self.device)
            c_0 = torch.zeros(1, self.N, self.H, dtype=torch.float).to(self.device)
            self.model.train()
            for i, (x, y) in enumerate(tqdm(data_loader)):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                output, h_0, c_0 = self.model(x, h_0, c_0)
                h_0 = h_0.detach()
                c_0 = c_0.detach()
                loss = self.criterion(output.view(-1, self.V), y.view(-1))
                loss.backward()
                self.optimizer.step()
                self.add_loss(loss.item())
                if i % interval == 0:
                    ppl = np.exp(self.get_avg_loss())
                    self.info["history_train"].append([epoch * len(data_loader) + i,ppl])
                    print(f"{epoch},{i}:ppl={ppl}")


            eval_ppl = eval_proxity()
            self.info["history_eval"].append(eval_ppl)
            print(f"{epoch}:ppl={eval_ppl}")

        history_train = np.array(self.info["history_train"])
        plt.plot(history_train[:,0],history_train[:,1])
        plt.savefig(f"{self.name}_history_train.png")
        plt.show()
        history_eval = self.info["history_eval"]
        plt.plot(list(range(1,len(history_eval)+1)),history_eval)
        plt.savefig(f"{self.name}_history_eval.png")
        plt.show()

