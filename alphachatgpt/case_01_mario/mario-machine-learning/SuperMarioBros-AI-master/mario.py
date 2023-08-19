from operator import eq
import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import random
import os
import csv

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
from config import Config

class Mario(Individual):
    def __init__(self,
                 config: Config,
                 chromosome: Optional[Dict[str, np.ndarray]] = None,
                 hidden_layer_architecture: List[int] = [12, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Union[int, float] = np.inf,
                 name: Optional[str] = None,
                 debug: Optional[bool] = False,
                 fitness: int = 0,
                 game_score: int = 0,
                 farthest_x: int = 0
                 ):
        self.视觉记忆 = dict()
        self.视觉静止时长 = 0
        if not 'max_static_duration' in config.Misc.__dict__:
            config.Misc.__dict__['max_static_duration']=5
        
        self.config = config

        self.lifespan = lifespan
        self.name = name
        self.debug = debug
        self._fitness = fitness  # Overall fitness
        
        self._frames_since_progress = 0  # Number of frames since Mario has made progress towards the goal
        self._frames = 0  # Number of frames Mario has been alive
        
        self.hidden_layer_architecture = self.config.NeuralNetwork.hidden_layer_architecture
        self.hidden_activation = self.config.NeuralNetwork.hidden_node_activation
        self.output_activation = self.config.NeuralNetwork.output_node_activation

        self.start_row, self.viz_width, self.viz_height = self.config.NeuralNetwork.input_dims

        if self.config.NeuralNetwork.encode_row:
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height
        # print(f'num inputs:{num_inputs}')
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Input Nodes
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden Layer Ndoes
        self.network_architecture.append(6)                        # 6 Outputs ['u', 'd', 'l', 'r', 'a', 'b']

        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation),
                                          chromosome = chromosome
                                         )
        
        self.is_alive = True
        self.x_dist = None
        self.game_score = game_score
        self.did_win = False
        # This is mainly just to "see" Mario winning
        self.allow_additional_time  = self.config.Misc.allow_additional_time_for_flagpole
        self.additional_timesteps = 0
        self.max_additional_timesteps = int(60*2.5)
        self._printed = False

        # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # index                            0  1     2       3      4  5  6  7  8
        self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.buttons_press_sum = 0
        self.farthest_x = farthest_x

        # 这里应该弄个变量 记录马里奥啥都不做持续的时间，时间越长数字越大，输入神经网络


    @property
    def fitness(self):
        return self._fitness

    @property
    def chromosome(self):
        pass

    def decode_chromosome(self):
        pass

    def encode_chromosome(self):
        pass

    # 根据配置文件中的lambda算法，计算个体评分（适应度），评分高的个体被保留下来
    def calculate_fitness(self):
        frames = self._frames
        distance = self.x_dist
        score = self.game_score
        press_sum = self.buttons_press_sum

        self._fitness = self.config.GeneticAlgorithm.fitness_func(frames, distance, score, self.did_win,press_sum)

    # 将整个游戏画面裁剪为我们需要的范围
    def set_input_as_array(self, ram, tiles) -> None:
        mario_row, mario_col = SMB.get_mario_row_col(ram)
        arr = []
        
        for row in range(self.start_row, self.start_row + self.viz_height):
            for col in range(mario_col, mario_col + self.viz_width):
                try:
                    t = tiles[(row, col)]
                    # 这里没有检测马里奥所在的DynamicTileType，而是在
                    # 下面的encode_row单独拿了一列记录马里奥位置
                    if isinstance(t, StaticTileType):
                        if t.value == 0:
                            arr.append(0)
                        else:
                            arr.append(1)
                    elif isinstance(t, EnemyType):
                        arr.append(-1)
                    else:
                        raise Exception("This should never happen")
                except:
                    t = StaticTileType(0x00)
                    arr.append(0) # Empty
        self.inputs_as_array[:self.viz_height*self.viz_width, :] = np.array(arr).reshape((-1,1))

        # 如果启动了马里奥高度重点标记
        # 检测马里奥高度在上下限范围内，相当于重点记录马里奥的高度信息
        if self.config.NeuralNetwork.encode_row:
            # Assign one-hot for mario row
            row = mario_row - self.start_row
            one_hot = np.zeros((self.viz_height, 1))
            if row >= 0 and row < self.viz_height:
                one_hot[row, 0] = 1
            self.inputs_as_array[self.viz_height*self.viz_width:, :] = one_hot.reshape((-1, 1))

    # 视觉中枢
    def 视觉中枢(self, tiles) -> bool:
        # 这个tiles有问题，蘑菇，马里奥，都看不见。只能看到敌人和阻挡
        if tiles==self.视觉记忆:
            self.视觉静止时长 += 1
        else:
            self.视觉记忆 = tiles
            self.视觉静止时长 = 0
            
        if self.视觉静止时长 >= self.config.Misc.max_static_duration*60:
            return False
        return True

    # 由smb_ai.py的_update调用
    def update(self, ram, tiles, buttons, ouput_to_buttons_map) -> bool:
        """
        The main update call for Mario.
        Takes in inputs of surrounding area and feeds through the Neural Network
        
        Return: True if Mario is alive
                False otherwise
        """
        if self.is_alive:
            self._frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            # Sliding down flag pole
            # if ram[0x001D] == 3:
            #     # 到达终点
            #     self.did_win = True
            #     if not self._printed and self.debug:
            #         name = 'Mario '
            #         name += f'{self.name}' if self.name else ''
            #         print(f'{name} won')
            #         self._printed = True
            #     if not self.allow_additional_time:
            #         self.is_alive = False
            #         return False
            
            # 如果我们更近了一步，重设记录
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += 1

            if self.allow_additional_time and self.did_win:
                self.additional_timesteps += 1

            # 时间奖励检测 2.5秒(因为did_win永远为false，所以永远用不到)
            if self.allow_additional_time and self.additional_timesteps > self.max_additional_timesteps:
                self.is_alive = False
                return False
            # 自从突破检测 30秒
            elif not self.did_win and self._frames_since_progress > 60*15:
                self.is_alive = False
                return False            
        else:
            return False

        # Did you fly into a hole?
        # 碰怪死亡检测，但是和0x0E、0xB5有有啥关系？
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        if not self.视觉中枢(tiles):
            self.is_alive = False
            return False

        # 只裁取部分画面作为输入
        self.set_input_as_array(ram, tiles)

        # 运算那些按键需要按下
        output = self.network.feed_forward(self.inputs_as_array)
        threshold = np.where(output > 0.5)[0]
        
        # 按目标按键
        self.buttons_to_press.fill(0)  # Clear
        for b in threshold:
            self.buttons_to_press[ouput_to_buttons_map[b]] = 1
            self.buttons_press_sum += 1

        return True
    
def save_mario(population_folder: str, individual_name: str, mario: Mario) -> None:
    # Make population folder if it doesnt exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # 如果不存在 settings.config 则创建
    if 'settings.config' not in os.listdir(population_folder):
        with open(os.path.join(population_folder, 'settings.config'), 'w') as config_file:
            config_file.write(mario.config._config_text_file)
    
    # Make a directory for the individual
    folder_dir = os.path.join(population_folder, 'generations')
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)  #新建文件夹

                    #'lifespan': mario.lifespan,
                    #'name': mario.name,
                    #'fitness': mario._fitness,
                    #'game_score': mario.game_score,
                    #'farthest_x': mario.farthest_x},
    save_data={'config':{},'net_params':mario.network.params}
    np.save(os.path.join(folder_dir, individual_name), save_data)
    
def load_mario(population_folder: str, individual_name: str, config: Optional[Config] = None) -> Mario:
    individual_dir = os.path.join(population_folder,individual_name)
    # 确保文件在文件路径中存在
    if not os.path.exists(individual_dir):
        raise Exception(f'{individual_name} not found inside {population_folder}')
    
    load_data = np.load(individual_dir,allow_pickle=True).item()
    return Mario(config=config,
                 chromosome=load_data['net_params']
                 )

def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))

    return (mean, median, std, _min, _max)

def save_stats(population: Population, fname: str):
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    frames = [individual._frames for individual in population.individuals]
    max_distance = [individual.farthest_x for individual in population.individuals]
    fitness = [individual.fitness for individual in population.individuals]
    wins = [sum([individual.did_win for individual in population.individuals])]

    trackers = [('frames', frames),
                ('distance', max_distance),
                ('fitness', fitness),
                ('wins', wins)
                ]

    stats = ['mean', 'median', 'std', 'min', 'max']

    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(fname, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',', lineterminator='\n')
        if not os.path.exists(fname):
            writer.writeheader()

        row = {}
        # Create a row to insert into csv
        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat

        # Write row
        writer.writerow(row)

def load_stats(path_to_stats: str, normalize: Optional[bool] = False):
    data = {}

    fieldnames = None
    trackers_stats = None
    trackers = None
    stats_names = None

    with open(path_to_stats, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = reader.fieldnames
        trackers_stats = [f.split('_') for f in fieldnames]
        trackers = set(ts[0] for ts in trackers_stats)
        stats_names = set(ts[1] for ts in trackers_stats)
        
        for tracker, stat_name in trackers_stats:
            if tracker not in data:
                data[tracker] = {}
            
            if stat_name not in data[tracker]:
                data[tracker][stat_name] = []

        for line in reader:
            for tracker in trackers:
                for stat_name in stats_names:
                    value = float(line['{}_{}'.format(tracker, stat_name)])
                    data[tracker][stat_name].append(value)
        
    if normalize:
        factors = {}
        for tracker in trackers:
            factors[tracker] = {}
            for stat_name in stats_names:
                factors[tracker][stat_name] = 1.0

        for tracker in trackers:
            for stat_name in stats_names:
                max_val = max([abs(d) for d in data[tracker][stat_name]])
                if max_val == 0:
                    max_val = 1
                factors[tracker][stat_name] = float(max_val)

        for tracker in trackers:
            for stat_name in stats_names:
                factor = factors[tracker][stat_name]
                d = data[tracker][stat_name]
                data[tracker][stat_name] = [val / factor for val in d]

    return data

def get_num_inputs(config: Config) -> int:
    _, viz_width, viz_height = config.NeuralNetwork.input_dims
    if config.NeuralNetwork.encode_row:
        num_inputs = viz_width * viz_height + viz_height
    else:
        num_inputs = viz_width * viz_height
    return num_inputs

# 计算所有基因的个数
def get_num_trainable_parameters(config: Config) -> int:
    num_inputs = get_num_inputs(config)
    hidden_layers = config.NeuralNetwork.hidden_layer_architecture
    num_outputs = 6  # U, D, L, R, A, B

    layers = [num_inputs] + hidden_layers + [num_outputs]
    num_params = 0
    for i in range(0, len(layers)-1):
        L      = layers[i]
        L_next = layers[i+1]
        num_params += L*L_next + L_next

    return num_params
