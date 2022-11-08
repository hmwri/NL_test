import torch
from common.trainer import Trainer
from tqdm import tqdm
import numpy as np

class LstmTrainer(Trainer):
    def __init__(self,model, criterion,optimizer,device=None):
        super().__init__(model, criterion,optimizer,device)

        self.H = self.model.n_hidden
        self.V = self.model.n_vocab

    def fit(self,epoch_num,data_loader,data_loader_eval):
        self.N = data_loader.batch_size

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
                ppl = np.exp(loss_sum / len(data_loader))
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
                if i % 100 == 0:
                    ppl = np.exp(self.get_avg_loss())
                    print(f"{epoch},{i}:ppl={ppl}")

            eval_ppl = eval_proxity()
            print(f"{epoch}:ppl={eval_ppl}")
