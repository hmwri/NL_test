import torch.nn as nn
import torch

x = torch.tensor([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]],dtype=torch.float)
l = nn.Linear(3,4)
print(l(x))