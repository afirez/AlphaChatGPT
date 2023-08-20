# 无梯度强化学习：使用 Numpy 进行神经进化！

我们能否在没有反向传播的情况下解决简单的强化学习环境？

## 神经进化

### 简介

如果我告诉你，你可以在训练神经网络时，从未计算过梯度，只使用正向传播，你会怎么想？这就是神经进化的魔力所在！而且，我将展示所有这些都可以轻松地只使用 Numpy 完成！在学习统计学时，你会了解许多关于基于梯度的方法，但不久前我读了一篇由优步 AI 团队撰写的非常有趣的文章，他们展示了一个简单的遗传算法在解决 Atari 游戏方面，在墙上时钟时间上竟然能与最复杂的基于梯度的强化学习方法相媲美。我在下面附上了文章链接，如果你对强化学习感兴趣，我强烈建议你阅读一下。

### 什么是神经进化？

首先，对于那些尚不了解的人，神经进化描述了将进化和/或遗传算法应用于训练神经网络的结构和/或权重的过程，作为一种无梯度的替代方法！在这里，我们将使用一个极其简单的神经进化案例，仅使用固定拓扑网络，并专注于优化权重和偏差。神经进化的过程可以通过四个基本步骤来定义，这些步骤会重复进行，直到达到收敛，起始于一组随机生成的网络。

1. 评估种群的适应性
2. 选择适应性最强的个体进行繁殖
3. 使用适应性最强网络的副本进行再生
4. 对网络权重引入正态分布的突变
   
哇，这看起来相当简单！让我们稍微解释一些术语：

- 适应性：这仅仅描述了网络在特定任务上的表现如何，并允许我们确定哪些网络进行繁殖。请注意，由于进化算法是一种非凸优化形式，因此可以与任何损失函数一起使用，而不受其可微性（或缺乏可微性）的限制。

- 突变：这可能是最容易理解的部分！为了让我们的子网络得以改进，我们必须对网络权重引入随机变化，通常是从均匀或正态分布中抽取的。突变可以有许多不同的形式：平移突变（将参数乘以随机数）、交换突变（将参数替换为随机数）、符号突变（改变参数的符号）等等。在这里，我们只会使用简单的加性突变，但在这方面有很大的创造空间！

### 神经进化的优势

我们还应该考虑神经进化建模的理论优势。首先，我们只需要使用网络的正向传播，因为我们只需要计算损失来确定哪些网络进行繁殖。这样做的影响是显而易见的，反向传播通常是最耗时的！其次，进化算法在足够迭代次数的情况下保证能找到损失表面的全局最小值，而凸梯度优化方法可能会陷入局部最小值。最后，更复杂的神经进化形式不仅允许我们优化网络的权重，还可以优化网络的结构本身！

### 为什么不总是使用神经进化呢？

这实际上是一个复杂的问题，但归根结底，当有足够的梯度信息可用时，精确梯度下降方法更有效。这意味着损失表面越凸，你就越希望使用类似 SGD 的解析方法，而不是遗传算法。因此，在监督环境中很少会使用遗传算法，因为通常有足够的梯度信息可用，传统的梯度下降方法会表现得非常好。然而，如果你在强化学习环境中工作，或者在具有不规则损失曲面或低凸性（如顺序 GAN）的情况下工作，那么神经进化提供了一个可行的替代方案！事实上，最近的许多研究发现，在这些设置中，按参数进行神经进化模型可以取得更好的效果。

## 现在让我们开始吧！

### 加载库

正如在介绍中所述，我们将尝试仅使用 numpy 来完成这个项目，只定义我们所需的辅助函数！是的，我知道，还加载了 gym，但仅用于环境 ;)

```python
import numpy as np
import gym
```

### 关于数据

我们将使用 gym 中经典的 CartPole 环境来测试我们的网络。目标是通过左右移动来看网络能够将杆保持站立的时间有多长。作为一个强化学习任务，神经进化方法应该很适合！我们的网络将以4个观测作为输入，并将左或右作为动作输出。

### 辅助函数

我们首先会定义一些辅助函数来设置我们的网络。首先是 relu 激活函数，我们将它作为隐藏层的激活函数，然后是 softmax 函数，用于网络的输出，以获取网络输出的概率估计！最后，我们需要定义一个函数，用于生成响应向量的 one-hot 编码，以便在计算分类交叉熵时使用。

```python
def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    x = np.exp(x - np.max(x))
    x[x == 0] = 1e-15
    return np.array(x / x.sum())
```


这两个函数会帮助我们在后续的代码中进行网络的激活函数计算。

### 定义我们的网络类

接下来是有趣的部分！首先，我们将为种群内的个体网络定义一个类。我们需要定义一个初始化方法，该方法会随机分配权重和偏差，并将网络结构作为输入；一个预测方法，以便在给定输入时获取概率；最后一个评估方法，它返回给定输入和响应的网络的分类交叉熵！再次强调，我们只会使用我们定义的函数或来自 numpy 的函数。注意，初始化方法也可以接受另一个网络作为输入，这是我们将在代际之间执行突变的方式！

```python
class NeuralNet():
    
    def __init__(self, n_units=None, copy_network=None, var=0.02, episodes=50, max_episode_length=200):
        # Testing if we need to copy a network
        if copy_network is None:
            # Saving attributes
            self.n_units = n_units
            # Initializing empty lists to hold matrices
            weights = []
            biases = []
            # Populating the lists
            for i in range(len(n_units)-1):
                weights.append(np.random.normal(loc=0,scale=1,size=(n_units[i],n_units[i+1])))
                biases.append(np.zeros(n_units[i+1]))
            # Creating dictionary of parameters
            self.params = {'weights':weights,'biases':biases}
        else:
            # Copying over elements
            self.n_units = copy_network.n_units
            self.params = {'weights':np.copy(copy_network.params['weights']),
                          'biases':np.copy(copy_network.params['biases'])}
            # Mutating the weights
            self.params['weights'] = [x+np.random.normal(loc=0,scale=var,size=x.shape) for x in self.params['weights']]
            self.params['biases'] = [x+np.random.normal(loc=0,scale=var,size=x.shape) for x in self.params['biases']]
            
    def act(self, X):
        # Grabbing weights and biases
        weights = self.params['weights']
        biases = self.params['biases']
        # First propgating inputs
        a = relu((X@weights[0])+biases[0])
        # Now propogating through every other layer
        for i in range(1,len(weights)):
            a = relu((a@weights[i])+biases[i])
        # Getting probabilities by using the softmax function
        probs = softmax(a)
        return np.argmax(probs)
        
    # Defining the evaluation method
    def evaluate(self, episodes, max_episode_length, render_env, record):
        # Creating empty list for rewards
        rewards = []
        # First we need to set up our gym environment
        env = gym.make('CartPole-v0')
        # Recording video if we need to 
        if record is True:
            env = gym.wrappers.Monitor(env, "recording")
        # Increasing max steps
        env._max_episode_steps = 1e20
        for i_episode in range(episodes):
            observation = env.reset()
            for t in range(max_episode_length):
                if render_env is True:
                    env.render()
                observation, _, done, _ = env.step(self.act(np.array(observation)))
                if done:
                    rewards.append(t)
                    break
        # Closing our environment
        env.close()
        # Getting our final reward
        if len(rewards) == 0:
            return 0
        else:
            return np.array(rewards).mean()

```

这个类将帮助我们创建和操作神经网络，以及评估它们在 CartPole 环境中的表现。我们定义了初始化方法来随机初始化网络参数，act 方法用于执行网络的前向传播并生成动作，evaluate 方法用于评估网络在环境中的表现。

### 定义我们的遗传算法类

最后，我们需要定义一个管理种群的类，执行神经进化的四个关键步骤！我们需要三个方法。首先，初始化方法创建一组随机网络并设置属性。接下来，我们需要一个适应性方法，给定一个输入，重复执行上述步骤：首先评估网络，然后选择最适合的，创建子网络，最后突变子网络！最后，我们需要一个预测方法，以便我们可以使用该类训练的最佳网络。让我们开始测试吧！

```python
class GeneticNetworks():
    
    # Defining our initialization method
    def __init__(self, architecture=(4,16,2), population_size=50, generations=500, render_env=True, record=False,
                 mutation_variance=0.02, verbose=False, print_every=1, episodes=10, max_episode_length=200):
        # Creating our list of networks
        self.networks = [NeuralNet(architecture) for _ in range(population_size)]
        self.population_size = population_size
        self.generations = generations
        self.mutation_variance = mutation_variance
        self.verbose = verbose
        self.print_every = print_every
        self.fitness = []
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.render_env = render_env
        self.record = record
        
    # Defining our fitting method
    def fit(self):
        # Iterating over all generations
        for i in range(self.generations):
            # Doing our evaluations
            rewards = np.array([x.evaluate(self.episodes, self.max_episode_length, self.render_env, self.record) for x in self.networks])
            # Tracking best score per generation
            self.fitness.append(np.max(rewards))
            # Selecting the best network
            best_network = np.argmax(rewards)
            # Creating our child networks
            new_networks = [NeuralNet(copy_network=self.networks[best_network], var=self.mutation_variance, max_episode_length=self.max_episode_length) for _ in range(self.population_size-1)]
            # Setting our new networks
            self.networks = [self.networks[best_network]] + new_networks
            # Printing output if necessary
            if self.verbose is True and (i % self.print_every == 0 or i == 0):
                print('Generation:', i + 1, '| Highest Reward:', rewards.max().round(1), '| Average Reward:', rewards.mean().round(1))
        
        # Returning the best network
        self.best_network = self.networks[best_network]
```

这个类将帮助我们设置并管理网络种群，执行神经进化的步骤，并训练出在 CartPole 环境中表现最好的网络。我们可以使用 fit 方法来开始训练过程。

### 测试我们的算法！

如上所述，我们将在 CartPole 问题上测试我们的网络，只使用一个包含16个节点的隐藏层，以及两个输出节点，表示向左或向右移动。我们还需要对许多回合进行平均，这样我们就不会意外地选择一个不好的网络进入下一代！我在一些试错之后选择了许多这些参数，因此你的情况可能会有所不同！此外，我们只会引入方差为0.05的突变，以免破坏网络的功能。

```python
from time import time
start_time = time()
genetic_pop = GeneticNetworks(architecture=(4,16,2),
                                population_size=64, 
                                generations=5,
                                episodes=15, 
                                mutation_variance=0.1,
                                max_episode_length=10000,
                                render_env=False,
                                verbose=True)
genetic_pop.fit()
print('Finished in',round(time()-start_time,3),'seconds')

```

上述代码将训练一个种群的网络，并输出每代中最高和平均奖励。这个示例中，我们进行了5代训练，每代使用15个回合进行评估。训练过程可能需要一些时间，取决于你的计算机性能。

### 初始随机网络

首先，让我们看看一个随机初始化的网络在这个任务上的表现。显然，这里没有任何策略，杆子几乎立即倒下。请忽略下面 GIF 中的光标，Gym 的录制在 Windows 下效果不佳！

![InitialNetwork](Numpy-Neuroevolution/InitialNetwork.gif)

```python
random_network = NeuralNet(n_units=(4,16,2))
random_network.evaluate(episodes=1, max_episode_length=int(1e10), render_env=True, record=False)
```

上述代码会创建一个随机初始化的网络，并在 CartPole 环境中进行一次评估。你可以尝试运行这段代码，观察随机网络的表现。由于是随机策略，你会看到杆子很快就会倒下。

### 经过5代训练...

在仅经过5代训练后，我们可以看到我们的网络几乎已经完全掌握了 CartPole 的技巧！而且仅仅花费了约30秒的训练时间！请注意，随着进一步的训练，网络将学会几乎100%的时间内将杆保持完全站立，但目前我们只关心速度，5代训练的时间相对较短！我们应该将这视为神经进化的威力的一个很好的例子。

![5 gen TrainedNetwork](Numpy-Neuroevolution/TrainedNetwork.gif)

``` python
# Lets observe our best network
genetic_pop.best_network.evaluate(episodes=3, max_episode_length=int(1e10), render_env=True, record=False)
```

上述代码会评估经过训练的最佳网络，让你观察它在 CartPole 环境中的表现。你可以尝试运行这段代码，看看经过5代训练的网络如何解决 CartPole 任务。你会发现，网络现在能够长时间地保持杆子的平衡。

### 接下来呢？

显然，我们未来可以添加许多东西来进一步研究神经进化的有效性。首先，研究不同突变操作（例如交叉）的效果将是很有趣的。

将来考虑切换到现代的深度学习平台，如TensorFlow或PyTorch，也是一个明智的想法。请注意，遗传算法是高度可并行化的，因为我们只需在单个设备上运行每个网络，执行一次前向传播。无需镜像权重或复杂的分布策略！因此，随着每增加一个处理单元，运行时间几乎呈线性下降。

最后，我们应该在不同的强化学习任务中探索神经进化，甚至是其他情况，例如在生成对抗网络或长序列LSTM网络中难以评估梯度的情况。

### 进一步阅读

如果你对神经进化及其应用感兴趣，Uber在其工程博客上有一个精彩的页面，展示了神经进化在强化学习中的现代优势：

[Uber Engineering Blog](https://www.uber.com/zh-US/blog/san-francisco/engineering/)

该项目的源代码可以在我的公共GitHub仓库中找到：

[gursky1/Numpy-Neuroevolution](https://github.com/gursky1/Numpy-Neuroevolution)

这个仓库包含了你在本文中使用的神经进化代码。你可以在那里找到更多关于神经进化的信息和示例。