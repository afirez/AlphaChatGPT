## 项目简介
以下是原版作者的信息。

If you want to see a YouTube video describing this at a high level, and showcasing what was learned, take a look [here](https://www.youtube.com/watch?v=CI3FRsSAa_U).

If you want to see my blog explaining how all of this works in great detail, go [here](https://chrispresso.github.io/AI_Learns_To_Play_SMB_Using_GA_And_NN).

我从[小鑫学渣](https://gitee.com/xiaoxinxuezha/SuperMarioBros-AI)那里获得了本项目，并做了大量代码的中文注解，删除了几个我发现的原作者的冗余代码，供大家学习。此外我还修改了原来臃肿的存档模式，换成了一种更简洁的模式，但也因此以前的存档都没法用了，你可以通过 存档转换器.bat 转换旧存档为新存档。

项目含有两个分支：

SuperMarioBros-AI-master		基于cpu的版本，效率低

SuperMarioBros-AI-master-pytorch	因为我对于卷积的研究，修改出的基于pytorch的版本，有需要的可以尝试一下。但因为我加入了太多神经，反而比上面的cpu慢了点。

### 启动方式：

1.进入项目文件夹，打开cmd输入以下指令安装必备python环境：

`pip install -r requirements.txt`

(注：需要python3.6 3.7 3.8版本，如果有anaconda则可以通过activate python37切换python版本)

2.启动训练

Windows系统下双击`运行.bat`就可以启动

可以通过修改bat中的`set file="save\测试"`字段更改启动的存档。

3.新建自己的训练

修改项目settings .config中的[Statistics]条目

比如修改`save/测试/stats.csv`为`save/我的测试/stats.csv`，有两个选项都要改。

然后运行 新建.bat

当前无法直接训练1-2，2-2等。
关卡选项仅为：
1-1    2-1    3-1    4-1    5-1    6-1    7-1    8-1

此外，源作者说它还能够学习：敌人与旗杆故障、跳墙和快速加速。都是通过修改目标lambda函数实现的。

以下是我翻译的详细操作指南：

## 详细操作指南：
- [安装指导](#安装指导)
- [命令行选项](#命令行选项)
  - [配置](#配置)
  - [加载个体们](#加载个体们)
  - [重演个体们](#重演个体们)
  - [禁止显示](#禁止显示)
  - [Debug](#debug)
  - [手动](#手动)
- [创建新种群](#创建新种群)
- [理解配置文件](#理解配置文件)
  - [神经网络](#神经网络)
  - [绘制](#绘制)
  - [统计](#统计)
  - [遗传算法](#遗传算法)
  - [基因突变](#基因突变)
  - [基因重组](#基因重组)
  - [挑选亲代](#挑选亲代)
  - [其它](#其它)
- [理解运行界面](#理解运行界面)
- [查看统计信息](#查看统计信息)
- [结果](#结果)

## 安装指导
你需要3.6、3.7、3.8版本的python。

1. 命令行中运行`pip install -r requirements.txt`，将安装三个必备的依赖。
2. Install the ROM
   - 这里可以阅读[免责声明](https://wowroms.com/en/disclaimer)
   - H游戏本体下载的地址[Super Mario Bros.](https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html) and click `download`.
3. 运行例如`python -m retro.import "/path/to/unzipped/super mario bros. (world).zip"`
    - 请确保游戏本体`super mario bros. (world).zip`的路径正确。
    - 如果命令行输出如下，则表示您成功了:
      ```
      Importing SuperMarioBros-Nes
      Imported 1 games
      ```
## 命令行选项
如果您希望查看所有命令选项，可以运行`python smb_ai.py -h`。

### 配置
- `'-c', '--config' FILE`。生成默认配置文件，如果您不小心删除了所有配置文件。

### 加载个体们
您有时需要加载个体。这可能是因为你的电脑崩溃了，想加载一部分上次训练产生的个体们。你可能想把不同群体的个体结合起来进行实验。无论是哪种情况，都必须通过*同时*指定以下两个参数：
- `--load-file FOLDER`. 指定您想加载个体存档所在的文件夹。
- `--load-inds INDS`. 这决定了要实际加载文件夹中的哪些个体。如果您想要一个范围，可以执行`[2,100]`，它将加载个体`best_ind_gen2，best_ind_gen3...best_ind_gen100`。如果您想指定某些，你可以使用`5,10,15,12901`，用英文逗号分隔，没有空格。此外作为翻译者，我还添加了`last`选项，它将加载最新的两代个体，省去了次次人工修改的麻烦。
请注意，加载个体只支持从某代中加载性能最好的个体。因为只有它被存档了。

### 重演个体们
如果你想观看特定个体重演他们的操作，这很有帮助。您必须*同时*指定以下两个参数：
- `--replay-file FOLDER`. 指定您想重演个体存档所在的文件夹。
- `--replay-inds INDS`. 它接受与`--load-inds`相同的语法，并添加了一个语法。如果您想从某个体开始重演到文件夹里的末尾，您可以执行`[12，]`，这将从“best_ind_gen12”开始并贯穿整个文件夹。*注*：`--load inds`中不支持这一点，因为当您加载时它们将被视为初始群体，作为亲代开始产生下一代。正因如此，您可能会意外地在你的群体基因库中拥有比预期多得多的亲代，或更先进的某代个体和过去落后的个体产生下一代。

### 禁止显示
可惜的是训练的速度被`Qt`屏幕刷新器限制了。因此，当显示器打开时（无论是否隐藏），您只能以显示器的刷新率运行。模拟器支持更快的运行，因此创建了这个仅通过命令行运行的选项，有助于全速训练速度。
- `--no-display`。当存在这个参数时，将不会存在画面窗口。

### Debug
如果您想知道种群在何时进步或个人何时获胜，可以设置debug标志。如果您禁用了显示器，但希望了解您种群的情况，这将很有帮助。
- `--debug`。当存在此选项时，有关辈分、获胜个体、新的最大距离等的某些信息将打印到命令行。

### 手动
本翻译者由于测试的需要添加的模式，手动操纵mario。比如用来检测新添加的功能是否如愿运行。
- `--manual`。如`python smb_ai.py --manual --load-file 存档位置 --load-inds 个体"`。

## 创建新种群
如果你想创造一个新的种群，这很容易。如果使用默认的`settings.config`，请确保更改`save_best_individeal_from_generation`和`save_population_stats`以确保您希望将信息保存在何处。一旦您有了所需的配置文件，只需运行`python smb_ai.py-c settings.config`就可以生成新的种群。

## 理解配置文件
配置文件用于指导系统如何初始化、绘图、存档位置、遗传算法参数等等。理解各个选项的含义和如何更改他们是非常重要的。

### 神经网络
定义为`[NeuralNetwork]`。
- `input_dims :Tuple[int, int, int]`. This defines the "pink box". 参数有`(start_row, width, height)`，`start_row`开始于`0`在屏幕顶部，随着它的增加将不断靠近屏幕底部；`width`定义了粉框从`start_row`开始有多宽。`height`定义了粉框有多高。现在我还没支持`start_col`，目前他的位置是mairo所在的高度。
- `hidden_layer_architecture :Tuple[int, ...]`。描述了有多少节点在各个隐藏层。如`(12, 9)` 表示第一层有12个神经节，第二层有9个神经节。至少有一层隐藏层。
- `hidden_node_activation :str`. 选项有：`(relu, sigmoid, linear, leaky_relu, tanh)`。确定了在隐藏层中应用什么激活函数。
- `output_node_activation :str`. 选项有：`(relu, sigmoid, linear, leaky_relu, tanh)`。确定了在输出层中应用什么激活函数。
- `encode_row :bool`. 是否通过额外的独热码专门标记mario的高度位置。

### 绘制
定义为`[Graphics]`。极度拉低了fps。
- `tile_size :Tuple[int, int]`. 绘制时每个釉面的像素大小。
- `neuron_radius :float`. 绘制时神经节点的半径。

### 统计
定义为`[Statistics]`。
- `save_best_individual_from_generation :str`. 一个文件夹`save/mario`，用于存放每回合的最佳个体，供下次读档。新建模式时。同时自动将配置文件复制在这里。
- `save_population_stats :str`. `save/mario/stats.csv` 存放历史数据的文件夹。

### 遗传算法
定义为`[GeneticAlgorithm]`。
- `fitness_func :lambda`. 这是个lambda函数，一个个体结束训练后用来计算它的适应分，接受如下类型：
~~~python
def fitness_func(frames, distance, game_score, did_win):
  """
frames :int : mario存货的时间
distance :int : 总共横向移动的距离
game_score :int : mario获得游戏分数，诸如击杀、硬币之类的。
did_win :bool : True/False 是否mario完成了本关卡。
"""
~~~
因为它是个lambda函数，所以唯一的一段代码就是return。这意味着没法使用if语句，所以我使用了`max`、`min`函数替代`if`语句的功能。他不能返回0，所以一般使用`max(你的函数,0.001)`返回一个大于0的结果。

### 基因突变
定义为`[Mutation]`。
- `mutation_rate :float`. 突变率，数值必须为 `[0.00, 1.00)`. 指定了每个*基因*发生突变的概率。在这种情况下，每个可训练的参数都是一个*基因*。
- `mutation_rate_type :str`. 突变类型，选项为 `(static, dynamic)`. `static` 静态突变意味着突变率总是保持不变；`dynamic` 动态突变意味着突变率随着代数增加而减少。
- `gaussian_mutation_scale :float`. 高斯突变规模，当一个突变发生时，它是一个正常的高斯突变`N(0，1)`，所以因为参数的上限在“[0.00,1.00)”之间。所以这里提供了一个参数来缩小这一范围。然后突变将是`N（0,1）*范围`。

### 基因重组
定义为`[Crossover]`。
- `probability_sbx :float`. 执行模拟二进制交叉（SBX）的概率。到目前为止只能是`1.0`，因为相关代码还未完善。
- `sbx_eta :int`. 这是一个有点复杂的参数，但值越小产生后代的方差就越大。随着参数的增加，方差减小，子代更多地以亲代值为中心。`100`时仍然存在差异，但更多地围绕着父母。这有助于基因能够逐渐改变，而不是突然改变。
- `crossover_selection :str`. 选项是`(roulette, tournament)`. `roulette`轮盘赌 将每个个体适应分加和，然后用每个个体适应分占加和的概率，作为它可能参加繁衍的概率。`tournament`锦标赛 从整体中随机选择n个个体，然后从这n个个体中选择适应分最大的个体。
- `tournament_size :int`. 如果`crossover_selection = tournament`才使用这个选项，用于控制选择个体的数量。

### 挑选亲代
定义为`[Selection]`。
- `num_parents :int`. 训练开始时的亲代数量
- `num_offspring :int`. 子代数量，也可以理解为最大数量。小于这个数量则在回合结束后补充到这个数目。
- `selection_type :str`. 选项是 `(comma, plus)`. 让我通过下面的例子解释一下，comma`(50, 100)` 和plus`(50 + 100)`的区别:
  - `(50, 100)` 最开始产生50个亲代，并在每回合结束后根据选择函数选择几个亲代通过基因重组和变异产生下一代，维持在总数100个。亲代被选择下来的会继续在下一回合发挥，这与原作者的不同。
  - `(50 + 100)` 最开始产生50个亲代，并在每回合结束后根据选择函数选择几个亲代通过基因重组和变异产生100个后代，并维持在总数150个。亲代会进入下一回合，直至生命结束(lifespan归零)。
- `lifespan :float`. 当然该是int，但是为了使用`inf`用了folat。规定了一个mario可以存活多少回合。只在`selection_type = plus`时有用。

### 其它
定义为`[Misc]`。
- `level :str`. 现在支持的选项有：`(1-1, 2-1, 3-1, 4-1, 5-1, 6-1, 7-1, 8-1)` 可以通过添加`state`信息进入`gym environment`支持更多。
- `allow_additional_time_for_flagpole :bool`. 从实际来说，maio摸到旗杆后就死了，转为一个走进城堡的动画。你可能希望通过调整这个给mario多留点时间让它进入1-2,2-2等关卡。

## 理解运行界面
这里作为翻译者简单介绍一下，窗口内，右面的是游戏实际画面，下面是详细信息；左面是右面的抽象，比如红色釉面代表了敌人，有个粉框框内的是实际神经网络输入，下面是神经网络的简单绘制。

此外我还在标题栏添加了fps显示，e键暂停。

可以通过ctrl+v隐藏上述信息，有助于加速演化。
	
## 查看统计信息
stats.csv文件包含关于“存活时长、距离、适应分、获胜”的“平均值、中位数、标准值、最小值、最大值”的信息。如果您想查看.csv的最大距离，可以执行以下操作：
~~~python
stats = load_stats('/path/to/stats.csv')
stats['distance']['max']
~~~
这里是一段如何使用`matplotlib.pyplot`绘制上述stats曲线的示例代码:
~~~python
from mario import load_stats
import matplotlib.pyplot as plt

stats = load_stats('/path/to/stats.csv')
tracker = 'distance'
stat_type = 'max'
values = stats[tracker][stat_type]

plt.plot(range(len(values)), values)
ylabel = f'{stat_type.capitalize()} {tracker.capitalize()}' 
plt.title(f'{ylabel} vs. Generation')
plt.ylabel(ylabel)
plt.xlabel('Generation')
plt.show()
~~~

## 结果
不同的mario个体在不同的配置中学习，下面是一些mario可以学到的例子：

Mario beating 1-1:
![Mario Beating 1-1](/SMB_level1-1_gen_258_win.gif)

Mario beating 4-1:
![Mario Beating 4-1](/SMB_level4-1_gen_1378_win.gif)

Mario learning to walljump:
![Mario Learning to Walljump](/SMB_level4-1_walljump.gif)