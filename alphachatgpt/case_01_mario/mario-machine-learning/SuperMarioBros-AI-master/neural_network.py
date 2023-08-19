import numpy as np
from typing import NewType,Callable,Tuple, Optional, Union, Set, Dict, Any, List
#from torch import nn

ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])

sigmoid = ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))
tanh = ActivationFunction(lambda X: np.tanh(X))
relu = ActivationFunction(lambda X: np.maximum(0, X))
leaky_relu = ActivationFunction(lambda X: np.where(X > 0, X, X * 0.01))
linear = ActivationFunction(lambda X: X)


class FeedForwardNetwork(object):
    def __init__(self,
                 layer_nodes: List[int],
                 hidden_activation: ActivationFunction,
                 output_activation: ActivationFunction,
                 init_method: Optional[str] = 'uniform',
                 seed: Optional[int] = None,

                 chromosome: Optional[Dict[str, np.ndarray]] = None
                 ):
        self.layer_nodes = layer_nodes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.inputs = None
        self.out = np.zeros(6)

        if chromosome:
            self.params = chromosome.copy()
            if not 'A' in list(self.params.keys())[-1]:
                for i in range(int(len(self.params)/2)):
                    self.params['A' + str(i+1)] = np.zeros(self.params['b'+str(i+1)].shape)
        else:
            self.params = {}
            self.rand = np.random.RandomState(seed)
            # 初始化权重和偏移
            for l in range(1, len(self.layer_nodes)):
                if init_method == 'uniform':
                    self.params['W' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], self.layer_nodes[l-1]))
                    self.params['b' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], 1))
                else:
                    raise Exception('Implement more options, bro')
                self.params['A' + str(l)] = None
        
        
    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        A_prev = X
        L = len(self.layer_nodes) - 1  # len(self.params) // 2

        # Feed hidden layers
        # 这样的话，A_prev在循环中一步步的被改变，就相当于是在层之间传递了
        for l in range(1, L):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_prev = self.hidden_activation(Z)
            self.params['A' + str(l)] = A_prev #存起来一会去画

        # Feed output 这里的out被其他程序调用,所以不能和上面的循环合并
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A_prev) + b
        out = self.output_activation(Z)
        self.params['A' + str(L)] = out

        self.out = out
        return out

    def softmax(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) / np.sum(np.exp(X), axis=0)

def get_activation_by_name(name: str) -> ActivationFunction:
    activations = [('relu', relu),
                   ('sigmoid', sigmoid),
                   ('linear', linear),
                   ('leaky_relu', leaky_relu),
                   ('tanh', tanh),
    ]

    func = [activation[1] for activation in activations if activation[0].lower() == name.lower()]
    assert len(func) == 1

    return func[0]
