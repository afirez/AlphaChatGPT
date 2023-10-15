import numpy as np
import matplotlib.pylab as plt

"""
evoluation weight

multi layer
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
        [0.0],
        [1.0],
        [1.0],
        [1.0],
        [0.0],
    ]
)

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

ins = 5
outs = 1
nodes = 5  # 15, 5

def weight(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws

w0 = weight(ins + 1, nodes)
ws = weight(nodes, outs)

errs = []
for i in range(5000):
    z = xs @ w0
    x = np.sin(z)
    yh = x @ ws 
    
    e = (yh - ys) * 1 #derivative of x=1
    ws -= (x.T @ e) * 0.01

    e = np.sum(np.abs(e))
    if e < 0.05:
        print("found solution")
        print(ws)

    errs.append(e)

plt.figure(1)
plt.plot(errs)
plt.show()