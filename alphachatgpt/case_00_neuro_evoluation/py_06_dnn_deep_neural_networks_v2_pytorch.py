import numpy as np
import matplotlib.pylab as plt

import torch
from torch.nn import functional as F

"""
evoluation weight

multi layer:
  1. 简单线性问题，使用简单感知机即可；
  2. 简单非线性问题， 使用一层隐藏层即可
  3. 复杂非线性问题， 使用深度神经网络

gradient descent
"""

xs = np.asarray(
    [
        [0,1,0,1,0],
        [0,0,1,1,0],
        [1,1,0,1,0],
        [1,1,1,0,1],
        [0,0,0,1,0],
    ]
)

# ws = np.ndarray([1,0,1,0,-1]) # hidden

ys = np.asarray(
    [
        [0],
        [0],
        [0],
        [3],
        [3],
    ]
)

# xs = np.asarray([[1,0], [0,1], [1,1], [0,0]])
# ys = np.asarray([[1], [1], [0], [0]])

xs = np.asarray([[-10], [-8], [-6], [-4], [-2], [0],[2], [4], [6], [8], [10]])
ys = 3 * xs - 2
ys = 0.5 * xs + 7
ys = xs ** 2
ys = xs ** 3 - xs ** 2 + xs - 3

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

xs = torch.tensor(xs).float()
ys = torch.tensor(ys).float()


ins = 1
outs = 1
nodes = 100   # 15, 5 , 2

lr = 0.03
lr = 0.0003
lr = 0.00003
lr = 0.000003
lr = 0.0000001
lr = 0.003


params = []

def weight(ins, outs):
    ws = torch.randn(ins, outs)
    ws = ws.requires_grad_(True)
    return ws

class Model():
    def __init__(self) -> None:
        self.w0 = weight(ins + 1, nodes)
        self.w1 = weight(nodes, nodes)
        self.w2 = weight(nodes, nodes)
        self.w3 = weight(nodes, outs)
        params.append(self.w0)
        params.append(self.w1)
        params.append(self.w2)
        params.append(self.w3)

    def forward(self, x):
        x = torch.sin(x @ self.w0)
        x = torch.sin(x @ self.w1)
        x = torch.sin(x @ self.w2)
        yh = (x @ self.w3) 
        return yh

model =  Model()

# optimiser = torch.optim.SGD([w0, w1, w2], lr)
optimiser = torch.optim.Adam(params, lr)

errs = []
for i in range(5000):
    yh = model.forward(xs)

    loss = F.mse_loss(yh, ys)
    optimiser.zero_grad()

    loss.backward()
    optimiser.step()

    e = loss.item()
    if i % 500 == 0:
        print(f"loss: {e}")

    errs.append(e)

    # x0 = xs

    # z0 = x0 @ w0; x1 = np.sin(z0)
    # z1 = x1 @ w1; x2 = np.sin(z1)
    # yh = x2 @ w2 

    # e = (yh - ys)

    # e2 = (e) * 1
    # e1 = (e2 @ w2.T) * np.cos(z1) 
    # e0 = (e1 @ w1.T) * np.cos(z0) 

    # w2 -= (x2.T @ e2) * lr
    # w1 -= (x1.T @ e1) * lr
    # w0 -= (x0.T @ e0) * lr 

    # e = np.sum(np.abs(e))

    # errs.append(e)

value = 2
value = torch.tensor([value,1]).float()
result = model.forward(value)
print(f"model.forward({value}) == {result}")

plt.figure(1)
plt.plot(errs)

plt.figure(2)
plt.plot(ys)
plt.plot(yh.detach())

plt.show()