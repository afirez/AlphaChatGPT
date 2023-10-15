import numpy as np
import matplotlib.pylab as plt

"""
random search weight
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
        [1],
        [1],
        [1],
        [0],
    ]
)

ins = 5
outs = 1

def weight(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws

ws = weight(ins, outs)

errs = []
for i in range(5000):
    yh = xs @ ws 
    e = yh - ys
    e = np.sum(np.abs(e))
    if e < 0.05:
        print("found solution")
        print(ws)
    else:
        ws = weight(ins, outs)
        
    errs.append(e)

plt.plot(errs)
plt.show()