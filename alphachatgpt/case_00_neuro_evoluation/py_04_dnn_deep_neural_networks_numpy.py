import numpy as np
import matplotlib.pylab as plt

"""
evoluation weight

multi layer:
  1. 简单线性问题，使用简单感知机即可；
  2. 简单非线性问题， 使用一层隐藏层即可
  3. 复杂非线性问题， 使用深度神经网络

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

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

ins = 1
outs = 1
nodes = 100   # 15, 5 , 2
lr = 0.03
lr = 0.0003
lr = 0.00003
lr = 0.000003

def weight(ins, outs):
    ws = np.random.randn(ins, outs) * 0.1
    return ws

w0 = weight(ins + 1, nodes)
w1 = weight(nodes, nodes)
w2 = weight(nodes, nodes)
w3 = weight(nodes, outs)

errs = []
for i in range(5000):
    x0 = xs

    z0 = x0 @ w0; x1 = np.sin(z0)
    z1 = x1 @ w1; x2 = np.sin(z1)
    z2 = x2 @ w2; x3 = np.sin(z2)
    yh = x3 @ w3 

    e = (yh - ys)

    e3 = (e) * 1  #derivative of x=1
    e2 = (e3 @ w3.T) * np.cos(z2) 
    e1 = (e2 @ w2.T) * np.cos(z1) 
    e0 = (e1 @ w1.T) * np.cos(z0) 

    w3 -= (x3.T @ e3) * lr
    w2 -= (x2.T @ e2) * lr
    w1 -= (x1.T @ e1) * lr
    w0 -= (x0.T @ e0) * lr 

    e = np.sum(np.abs(e))

    errs.append(e)

plt.figure(1)
plt.plot(errs)

plt.figure(2)
plt.plot(ys)
plt.plt(yh.detach())

plt.show()