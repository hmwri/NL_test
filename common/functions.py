import torch
import numpy as np
def eval_proxity(model,criterion,dataloader,vocab_size):
    loss_sum = 0
    hidden1, hidden2 = None, None

    # 勾配を計算しないモードへ
    with torch.no_grad():
        # モデルを評価モードへ
        model.eval()
        for x,t in dataloader:
            output, (hidden1, hidden2) = model(x, hidden1, hidden2)
            loss = criterion(output.view(-1, vocab_size), t.view(-1))
            loss_sum += loss.item()
        ppl = np.exp(loss_sum / len(dataloader))
        return ppl

