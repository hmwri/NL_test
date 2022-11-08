import torch

class Trainer:
    def __init__(self,model, criterion,optimizer,device=None):
        if device is "cpu":
            pass
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        device = torch.device(device)
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = 0
        self.loss_count = 0

    def add_loss(self,loss):
        self.loss += loss
        self.loss_count += 1

    def get_avg_loss(self):
        avg =  self.loss / self.loss_count
        self.loss = 0
        self.loss_count = 0
        return avg


