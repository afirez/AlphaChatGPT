import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from PIL import Image
from PIL.ImageQt import ImageQt
from typing import Tuple, List, Optional
import random
import sys
import math
import numpy as np
import argparse
import os
import gzip
import time


from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config
from nn_viz import NeuralNetworkViz
from mario import Mario, save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation

normal_font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)


def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)


class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config, nn_viz: NeuralNetworkViz):
        super().__init__(parent)
        self.size = size
        self.config = config
        self.nn_viz = nn_viz
        self.ram = None
        self.x_offset = 150
        self.tile_width, self.tile_height = self.config.Graphics.tile_size
        self.tiles = None
        self.enemies = None
        self._should_update = True

    #绘制游戏区
    def _draw_region_of_interest(self, painter: QPainter) -> None:
        # Grab mario row/col in our tiles
        mario = SMB.get_mario_location_on_screen(self.ram)
        mario_row, mario_col = SMB.get_mario_row_col(self.ram)
       
        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))

        start_row, viz_width, viz_height = self.config.NeuralNetwork.input_dims
        painter.drawRect(mario_col*self.tile_width + 5 + self.x_offset, start_row*self.tile_height + 5, viz_width*self.tile_width, viz_height*self.tile_height)

    #绘制釉面区
    def draw_tiles(self, painter: QPainter):
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                x_start = 5 + (self.tile_width * col) + self.x_offset
                y_start = 5 + (self.tile_height * row)

                tile = self.tiles[(row, col)]
                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
                    color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))

                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        if self._should_update:
            draw_border(painter, self.size)
            if not self.ram is None:
                self.draw_tiles(painter)
                self._draw_region_of_interest(painter)
                self.nn_viz.show_network(painter)
        else:
            # draw_border(painter, self.size)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
            txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
            painter.drawText(event.rect(), Qt.AlignCenter, txt)
            pass

        painter.end()

    def _update(self):
        self.update()


class GameWindow(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config):
        super().__init__(parent)
        self._should_update = True
        self.size = size
        self.config = config
        self.screen = None
        self.img_label = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.img_label)
        self.setLayout(self.layout)
        

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        if self._should_update:
            draw_border(painter, self.size)
            if not self.screen is None:
                # self.img_label = QtWidgets.QLabel(self.centralWidget)
                # screen = self.env.reset()
    
                width = self.screen.shape[0] * 3 
                height = int(self.screen.shape[1] * 2)
                resized = self.screen
                original = QImage(self.screen, self.screen.shape[1], self.screen.shape[0], QImage.Format_RGB888)
                # Create the image and label
                qimage = QImage(original)
                # Center where the image will go
                x = (self.screen.shape[0] - width) // 2
                y = (self.screen.shape[1] - height) // 2
                self.img_label.setGeometry(0, 0, width, height)
                # Add image
                pixmap = QPixmap(qimage)
                pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
                self.img_label.setPixmap(pixmap)
        else:
            self.img_label.clear()
            # draw_border(painter, self.size)
        painter.end()

    def _update(self):
        self.update()

#信息组件类
class InformationWidget(QtWidgets.QWidget):
    def __init__(self, parent, size, config):
        super().__init__(parent)
        self.size = size
        self.config = config

        self.grid = QtWidgets.QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self._init_window()
        # self.grid.setSpacing(20)
        self.setLayout(self.grid)
        

    def _init_window(self) -> None:
        info_vbox = QVBoxLayout()
        info_vbox.setContentsMargins(0, 0, 0, 0)
        ga_vbox = QVBoxLayout()
        ga_vbox.setContentsMargins(0, 0, 0, 0)

        # 当前辈份
        generation_label = QLabel()
        generation_label.setFont(font_bold)
        generation_label.setText('辈份:')
        generation_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.generation = QLabel()
        self.generation.setFont(normal_font)
        self.generation.setText("<font color='red'>" + '1' + '</font>')
        self.generation.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_generation = QHBoxLayout()
        hbox_generation.setContentsMargins(5, 0, 0, 0)
        hbox_generation.addWidget(generation_label, 1)
        hbox_generation.addWidget(self.generation, 1)
        info_vbox.addLayout(hbox_generation)

        # 当前个体
        current_individual_label = QLabel()
        current_individual_label.setFont(font_bold)
        current_individual_label.setText('个体:')
        current_individual_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.current_individual = QLabel()
        self.current_individual.setFont(normal_font)
        if args.load_inds:
            self.current_individual.setText('1/{}'.format(len(args.load_inds)))
        elif args.config:
            self.current_individual.setText('1/{}'.format(self.config.Selection.num_offspring))
        else:
            self.current_individual.setText('1/{}'.format(len(args.replay_inds)))
        self.current_individual.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_current_individual = QHBoxLayout()
        hbox_current_individual.setContentsMargins(5, 0, 0, 0)
        hbox_current_individual.addWidget(current_individual_label, 1)
        hbox_current_individual.addWidget(self.current_individual, 1)
        info_vbox.addLayout(hbox_current_individual)

        # 生存适应分
        best_fitness_label = QLabel()
        best_fitness_label.setFont(font_bold)
        best_fitness_label.setText('最优适应分:')
        best_fitness_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.best_fitness = QLabel()
        self.best_fitness.setFont(normal_font)
        self.best_fitness.setText('0')
        self.best_fitness.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_best_fitness = QHBoxLayout()
        hbox_best_fitness.setContentsMargins(5, 0, 0, 0)
        hbox_best_fitness.addWidget(best_fitness_label, 1)
        hbox_best_fitness.addWidget(self.best_fitness, 1)
        info_vbox.addLayout(hbox_best_fitness) 

        # 最远距离
        max_distance_label = QLabel()
        max_distance_label.setFont(font_bold)
        max_distance_label.setText('最远距离:')
        max_distance_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.max_distance = QLabel()
        self.max_distance.setFont(normal_font)
        self.max_distance.setText('0')
        self.max_distance.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_max_distance = QHBoxLayout()
        hbox_max_distance.setContentsMargins(5, 0, 0, 0)
        hbox_max_distance.addWidget(max_distance_label, 1)
        hbox_max_distance.addWidget(self.max_distance, 1)
        info_vbox.addLayout(hbox_max_distance)

        # 神经网络输入数
        num_inputs_label = QLabel()
        num_inputs_label.setFont(font_bold)
        num_inputs_label.setText('网络输入数:')
        num_inputs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        num_inputs = QLabel()
        num_inputs.setFont(normal_font)
        num_inputs.setText(str(get_num_inputs(self.config)))
        num_inputs.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_num_inputs = QHBoxLayout()
        hbox_num_inputs.setContentsMargins(5, 0, 0, 0)
        hbox_num_inputs.addWidget(num_inputs_label, 1)
        hbox_num_inputs.addWidget(num_inputs, 1)
        info_vbox.addLayout(hbox_num_inputs)

        # 基因个数
        trainable_params_label = QLabel()
        trainable_params_label.setFont(font_bold)
        trainable_params_label.setText('基因个数:')
        trainable_params_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        trainable_params = QLabel()
        trainable_params.setFont(normal_font)
        trainable_params.setText(str(get_num_trainable_parameters(self.config)))
        trainable_params.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_trainable_params = QHBoxLayout()
        hbox_trainable_params.setContentsMargins(5, 0, 0, 0)
        hbox_trainable_params.addWidget(trainable_params_label, 1)
        hbox_trainable_params.addWidget(trainable_params, 1)
        info_vbox.addLayout(hbox_trainable_params)

        # Selection
        selection_type = self.config.Selection.selection_type
        num_parents = self.config.Selection.num_parents
        num_offspring = self.config.Selection.num_offspring
        if selection_type == 'comma':
            selection_txt = '{}, {}'.format(num_parents, num_offspring)
        elif selection_type == 'plus':
            selection_txt = '{} + {}'.format(num_parents, num_offspring)
        else:
            raise Exception('Unkown Selection type "{}"'.format(selection_type))
        selection_hbox = self._create_hbox('亲子比:', font_bold, selection_txt, normal_font)
        ga_vbox.addLayout(selection_hbox)

        # 阳寿
        lifespan = self.config.Selection.lifespan
        lifespan_txt = '无限' if lifespan == np.inf else str(int(lifespan))
        lifespan_hbox = self._create_hbox('阳寿:', font_bold, lifespan_txt, normal_font)
        ga_vbox.addLayout(lifespan_hbox)

        # 突变率
        mutation_rate = self.config.Mutation.mutation_rate
        mutation_type = self.config.Mutation.mutation_rate_type.capitalize()
        mutation_txt = '{} {}% '.format(mutation_type, str(round(mutation_rate*100, 2)))
        mutation_hbox = self._create_hbox('突变率:', font_bold, mutation_txt, normal_font)
        ga_vbox.addLayout(mutation_hbox)

        # 筛选模式
        crossover_selection = self.config.Crossover.crossover_selection
        if crossover_selection == 'roulette':
            crossover_txt = '轮盘赌'
        elif crossover_selection == 'tournament':
            crossover_txt = '锦标赛({})'.format(self.config.Crossover.tournament_size)
        elif crossover_selection == 'elitism':
            crossover_txt = '精英主义'
        else:
            raise Exception('Unknown crossover selection "{}"'.format(crossover_selection))
        crossover_hbox = self._create_hbox('筛选模式:', font_bold, crossover_txt, normal_font)
        ga_vbox.addLayout(crossover_hbox)

        # 重组颗粒度
        sbx_eta_txt = str(self.config.Crossover.sbx_eta)
        sbx_hbox = self._create_hbox('重组颗粒度:', font_bold, sbx_eta_txt, normal_font)
        ga_vbox.addLayout(sbx_hbox)

        # 层数
        num_inputs = get_num_inputs(self.config)
        hidden = self.config.NeuralNetwork.hidden_layer_architecture
        num_outputs = 6
        L = [num_inputs] + hidden + [num_outputs]
        layers_txt = '[' + ', '.join(str(nodes) for nodes in L) + ']'
        layers_hbox = self._create_hbox('层数:', font_bold, layers_txt, normal_font)
        ga_vbox.addLayout(layers_hbox)

        self.grid.addLayout(info_vbox, 0, 0)
        self.grid.addLayout(ga_vbox, 0, 1)

    def _create_hbox(self, title: str, title_font: QtGui.QFont,
                     content: str, content_font: QtGui.QFont) -> QHBoxLayout:
        title_label = QLabel()
        title_label.setFont(title_font)
        title_label.setText(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        content_label = QLabel()
        content_label.setFont(content_font)
        content_label.setText(content)
        content_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(5, 0, 0, 0)
        hbox.addWidget(title_label, 1)
        hbox.addWidget(content_label, 1)
        return hbox


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        global args
        self.config = config
        self.top = 150
        self.left = 150
        self.width = 1100
        self.height = 700
        self.stop = False

        self.title = 'Super Mario Bros AI'
        self.current_generation = 0
        # This is the generation that is actual 0. If you load individuals then you might end up starting at gen 12, in which case
        # gen 12 would be the true 0
        # 在读档演化模式中，先展示一下亲代的本事，这个是亲代数量
        self._true_zero_gen = 0

        self._should_display = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)

        # 初始化当前帧操作列表
        # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
        # index                0  1     2       3      4  5  6  7  8
        self.keys = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
        # We need a mapping from the output to the keys above
        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }
        '''if args.manual:
            self.env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
            self.init_window()
            return'''
        self.env = retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.Misc.level}')

        # 初始化种群表
        individuals: List[Individual] = []

        # 加载所有个体到args.load_inds中
        num_loaded = 0
        if args.load_inds:
            # Overwrite the config file IF one is not specified
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.load_file, 'settings.config'))
                except:
                    raise Exception(f'settings.config not found under {args.load_file}')
                
            set_of_inds = set(args.load_inds)
            gen_dir = os.path.join(args.load_file, 'generations')
            for ind_name in os.listdir(gen_dir):
                if ind_name.startswith('best_ind_gen'):
                    # ind_number = int(ind_name[len('best_ind_gen'):-len('')])
                    ind_number = int(ind_name.replace('best_ind_gen','').replace('.npy',''))
                    if ind_number in set_of_inds:
                        individual = load_mario(gen_dir, ind_name, self.config)
                        # Set debug stuff if needed
                        if args.debug:
                            individual.name = f'm{num_loaded}_loaded'
                            individual.debug = True
                        individuals.append(individual)
                        num_loaded += 1
            # Set the generation
            self.current_generation = max(set_of_inds)  # +1 becauase it's the next generation
            self._true_zero_gen = self.current_generation

        # Load any individuals listed in args.replay_inds
        if args.replay_inds:
            # Overwrite the config file IF one is not specified
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.replay_file, 'settings.config'))
                except:
                    raise Exception(f'settings.config not found under {args.replay_file}')

            for ind_gen in args.replay_inds:
                ind_name = f'best_ind_gen{ind_gen}'
                fname = os.path.join(args.replay_file, ind_name)
                if os.path.exists(fname):
                    individual = load_mario(args.replay_file, ind_name, self.config)
                    # Set debug stuff if needed
                    if args.debug:
                        individual.name= f'm_gen{ind_gen}_replay'
                        individual.debug = True
                    individuals.append(individual)
                else:
                    raise Exception(f'No individual named {ind_name} under {args.replay_file}')
        '''# 如果是演化模式，我们需要填充个体到num_parents这么多
        else:
            num_parents = max(self.config.Selection.num_offspring - num_loaded, 0)
            # 如果等于0，那么是新建模式
            if num_loaded == 0:
                for _ in range(num_parents):
                    individual = Mario(self.config)
                    # Set debug stuff if needed
                    if args.debug:
                        individual.name = f'm{num_loaded}'
                        individual.debug = True
                    individuals.append(individual)
                    num_loaded += 1
            # 如果不等于0，就是读档模式
            else:
                if num_loaded == 1:  # 父母数量必须>1，否则复制一个自己
                    individuals.append(individuals[0])
                    num_loaded += 1
                self.population = Population(individuals)
                springs = self.produce_offspring(num_parents)
                num_loaded += len(springs)
                individuals.extend(springs)'''
        # 新建模式
        if args.config:
            for _ in range(self.config.Selection.num_offspring):
                individual = Mario(self.config)
                # Set debug stuff if needed
                if args.debug:
                     individual.name = f'm{num_loaded}'
                     individual.debug = True
                individuals.append(individual)
                num_loaded += 1
            # Set the generation
            self.current_generation = 1
            self._true_zero_gen = self.current_generation

        self.best_fitness = 0.0
        self._current_individual = 0
        self.population = Population(individuals)

        self.mario = self.population.individuals[self._current_individual]
        
        self.max_distance = 0  # Track farthest traveled in level
        self.max_fitness = 0.0
                
        # Determine the size of the next generation based off selection type
        self._next_gen_size = None
        if self.config.Selection.selection_type == 'plus':
            self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
        elif self.config.Selection.selection_type == 'comma':
            self._next_gen_size = self.config.Selection.num_offspring

        # If we aren't displaying we need to reset the environment to begin with
        if args.no_display:
            self.env.reset()
        else:
            self.init_window()

            # Set the generation in the label if needed
            if args.load_inds:
                txt = "<font color='red'>" + str(self.current_generation) + '</font>'  # +1 because we switch from 0 to 1 index
                self.info_window.generation.setText(txt)

            # if this is a replay then just set current_individual to be 'replay' and set generation
            if args.replay_file:
                self.info_window.current_individual.setText('Replay')
                txt = f"<font color='red'>{args.replay_inds[self._current_individual]}</font>"
                self.info_window.generation.setText(txt)

            self.show()

        self.fps_计数器=0
        self.实际fps=0
        self.fps_更新时间=time.time()
        # 设置帧率,开始循环
        if args.no_display:
            self._timer.start(1000 // 1000)
        elif 'fps' in config.Misc.__dict__:
            self._timer.start(1000 // config.Misc.fps)
        else:
            self._timer.start(1000 // 60)

    def init_window(self) -> None:
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.game_window = GameWindow(self.centralWidget, (514, 480), self.config)
        self.game_window.setGeometry(QRect(1100-514, 0, 514, 480))
        self.game_window.setObjectName('game_window')
        # 重开游戏并且传递游戏画面
        self.game_window.screen = self.env.reset()

        # 神经网络窗格
        self.viz = NeuralNetworkViz(self.centralWidget, self.mario, (1100-514, 700), self.config)
        self.viz_window = Visualizer(self.centralWidget, (1100-514, 700), self.config, self.viz)
        self.viz_window.setGeometry(0, 0, 1100-514, 700)
        self.viz_window.setObjectName('viz_window')
        self.viz_window.ram = self.env.get_ram()
        
        self.info_window = InformationWidget(self.centralWidget, (514, 700-480), self.config)
        self.info_window.setGeometry(QRect(1100-514, 480, 514, 700-480))

    def keyPressEvent(self, event):
        k = event.key()
        # print("按键代码："+str(k),Qt.Key_Z,Qt.Key_X,Qt.Key_C,Qt.Key_V)
        m = {
            Qt.Key_Z : 0,
            Qt.Key_X: 1,
            Qt.Key_C : 2,
            Qt.Key_V: 3,
            Qt.Key_B: 4,
            Qt.Key_Down: 5,
            Qt.Key_Left: 6,
            Qt.Key_Right : 7,
            Qt.Key_Up: 8
        }
        if k in m:
            self.keys[m[k]] = 1
        # print(self.keys)
        # 按下S键保存当前状态
        if k == Qt.Key_S:
            content = self.env.em.get_state()
            with gzip.open("the_saved_state.state", 'wb') as f:
                f.write(content)
            f.close()
            print("save the state")
        # ESC关闭程序
        if k == Qt.Key_Escape:
            sys.exit(app.exec_())
        elif k == Qt.Key_E:
            self.stop = not self.stop
            if self.stop:
                self._timer.stop()
            else:
                if args.no_display:
                    self._timer.start(1000 // 1000)
                elif 'fps' in self.config.Misc.__dict__:
                    self._timer.start(1000 // self.config.Misc.fps)
                else:
                    self._timer.start(1000 // 60)
                
            
        # ctrl+v隐藏游戏画面(运行速度直接提升两倍)
        modifier = int(event.modifiers())
        if modifier == Qt.CTRL:
            k = event.key()
            if k == Qt.Key_V:
                self._should_display = not self._should_display

    def keyReleaseEvent(self, event):
        k = event.key()
        m = {
            Qt.Key_Z : 0,
            Qt.Key_X: 1,
            Qt.Key_C : 2,
            Qt.Key_V: 3,
            Qt.Key_B: 4,
            Qt.Key_Down: 5,
            Qt.Key_Left: 6,
            Qt.Key_Right : 7,
            Qt.Key_Up: 8
        }
        if k in m:
            self.keys[m[k]] = 0

    def produce_offspring(self, num_spring :int) -> list():
        next_pop = []
        selection = self.config.Crossover.crossover_selection

        for i in range(num_spring):
            # 锦标赛模式，选出最优的当父母
            if selection == 'tournament':
                p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
            # 轮盘赌模式，被选出当父母的概率为 自己的适应度/所有个体适应度之和
            elif selection == 'roulette':
                p1, p2 = roulette_wheel_selection(self.population, 2)
            # 精英主义模式，没有随机，就是挑选最强的(会过拟合(偏执))
            elif selection == 'elitism':
                p1, p2 = elitism_selection(self.population, 2)
            else:
                raise Exception('crossover_selection "{}" is not supported'.format(selection))

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                # 亲代权重
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]
                # 亲代偏移
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # 重组
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # 变异
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)
                
                # 从重组/变异中分配子代参数
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l
                
                # 限制子代参数的最大最小值
                if True:
                    np.clip(c1_W_l, -1, 1, out=c1_params['W' + str(l)])
                    np.clip(c2_W_l, -1, 1, out=c2_params['W' + str(l)])
                    np.clip(c1_b_l, -1, 1, out=c1_params['b' + str(l)])
                    np.clip(c2_b_l, -1, 1, out=c2_params['b' + str(l)])

            # 产生子代
            c1 = Mario(self.config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
            c2 = Mario(self.config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

            # Set debug if needed
            if args.debug:
                c1_name = f'm{num_loaded}_new'
                c1.name = c1_name
                c1.debug = True
                num_loaded += 1

                c2_name = f'm{num_loaded}_new'
                c2.name = c2_name
                c2.debug = True
                num_loaded += 1
            next_pop.extend([c1, c2])
        return next_pop
        
    def next_generation(self) -> None:
        # Calculate fitness
        # print(', '.join(['{:.2f}'.format(i.fitness) for i in self.population.individuals]))

        if args.debug:
            print(f'----Current Gen: {self.current_generation}, True Zero: {self._true_zero_gen}')
            fittest = self.population.fittest_individual
            print(f'Best fitness of gen: {fittest.fitness}, Max dist of gen: {fittest.farthest_x}')
            num_wins = sum(individual.did_win for individual in self.population.individuals)
            pop_size = len(self.population.individuals)
            print(f'Wins: {num_wins}/{pop_size} (~{(float(num_wins)/pop_size*100):.2f}%)')

        if self.config.Statistics.save_best_individual_from_generation and self.current_generation != self._true_zero_gen:
            folder = self.config.Statistics.save_best_individual_from_generation
            best_ind_name = 'best_ind_gen{}'.format(self.current_generation)
            best_ind = self.population.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        if self.config.Statistics.save_population_stats and self.current_generation != self._true_zero_gen:
            fname = self.config.Statistics.save_population_stats
            save_stats(self.population, fname)

        self._increment_generation()
        self._current_individual = 0
        if not args.no_display:
            self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, self._next_gen_size))

        # 从所有个体中剔除到只剩 num_parents 个
        self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)

        random.shuffle(self.population.individuals)
        next_pop = []

        # Parents + offspring
        # 亲代+子代模式下，每次产生一些子代，亲代根据lifespan决定还能活多久
        if self.config.Selection.selection_type == 'plus':
            for individual in self.population.individuals:
                individual.lifespan -= 1 # 减少阳寿
                
                config = individual.config
                chromosome = individual.network.params
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan
                name = individual.name

                # 如果这个个体还有阳寿，就加入下一种群
                if lifespan > 0:
                    # 这里是不是作者傻了？直接next_pop.append(individual)不行吗？
                    m = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, lifespan)
                    # Set debug if needed
                    if args.debug:
                        m.name = f'{name}_life{lifespan}'
                        m.debug = True
                    next_pop.append(m)
        # 如果不是plus模式，那么就添加新个体直至不再少于_next_gen_size
        if len(next_pop) < self._next_gen_size:
            next_pop.extend(self.produce_offspring(self._next_gen_size - len(next_pop)))

        # Set next generation
        #random.shuffle(next_pop)
        self.population.individuals = next_pop

    # 执行遗传算法的交叉函数
    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eta = self.config.Crossover.sbx_eta

        # 重组权重和偏移
        child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
        child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, eta)

        return child1_weights, child2_weights, child1_bias, child2_bias

    # 执行遗传算法的变异函数
    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        mutation_rate = self.config.Mutation.mutation_rate
        scale = self.config.Mutation.gaussian_mutation_scale

        if self.config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)
        
        # Mutate weights
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

        # Mutate bias
        gaussian_mutation(child1_bias, mutation_rate, scale=scale)
        gaussian_mutation(child2_bias, mutation_rate, scale=scale)

    # 仅仅被上面的next_generation函数调用
    def _increment_generation(self) -> None:
        self.current_generation += 1
        if not args.no_display:
            txt = "<font color='red'>" + str(self.current_generation) + '</font>'
            self.info_window.generation.setText(txt)


    def _update(self) -> None:
        """
        这是主更新函数，调用频率取决于FPS计时器(FPS timer)。
        遗产算法的更新，窗口更新，诸如此类。
        """
        if time.time()-1>=self.fps_更新时间:
            self.fps_更新时间=time.time()
            self.实际fps=self.fps_计数器
            self.setWindowTitle('Super Mario Bros AI fps:'+str(self.fps_计数器))
            self.fps_计数器=0
        else:
            self.fps_计数器+=1
        
        # 让游戏虚拟机执行神经网络的操作，并获得结果
        # ret游戏画面 ram游戏内存
        if args.manual:
            ret = self.env.step(self.keys)
        else:
            ret = self.env.step(self.mario.buttons_to_press)
        ram=self.env.get_ram()
        tiles = SMB.get_tiles(ram)  # Grab tiles on the screen
        enemies = SMB.get_enemy_locations(ram)
        
        # 如果前台运行,则更新 游戏窗口、信息窗口、神经绘制窗口、神经网络窗格
        if not args.no_display:
            if self._should_display:
                self.game_window.screen = ret[0]
                self.game_window._should_update = True # 上面有个if，只有True时game_window才更新
                # 通过调用show函数直接使用构造函数完成实时信息更新
                self.info_window.show()
                # 更新神经窗格
                self.viz_window.ram = ram
                self.viz_window.tiles = tiles
                self.viz_window.enemies = enemies
                self.viz_window._should_update = True
            else:
                self.game_window._should_update = False
                self.viz_window._should_update = False
                self.info_window.hide()
            self.game_window._update()
            self.viz_window._update()   # 注释掉这个运行速度直接翻倍

        '''print('Player_X_Postion_In_Level',ram[SMB.RAMLocations.Player_X_Postion_In_Level.value])
        print('Player_X_Position_On_Screen',ram[SMB.RAMLocations.Player_X_Position_On_Screen.value])
        print('Player_X_Position_Screen_Offset',ram[SMB.RAMLocations.Player_X_Position_Screen_Offset.value])
        print('Player_Y_Position_Screen_Offset',ram[SMB.RAMLocations.Player_Y_Position_Screen_Offset.value])
        print('Player_Y_Pos_On_Screen',ram[SMB.RAMLocations.Player_Y_Pos_On_Screen.value])
        print('Player_Vertical_Screen_Position',ram[SMB.RAMLocations.Player_Vertical_Screen_Position.value])
        print('################################################')'''
        self.mario.update(ram, tiles, self.keys, self.ouput_to_keys_map)
        if args.manual:
            return
        
        if self.mario.is_alive:
            # 如果产生了新的最远距离，则用马里奥类的最远距离更新本类的最远距离
            if self.mario.farthest_x > self.max_distance:
                self.max_distance = self.mario.farthest_x
                if args.debug:
                    print('New farthest distance:', self.max_distance)
                # 如果前台运行，则更新最远距离的显示
                elif not args.no_display:
                    self.info_window.max_distance.setText(str(self.max_distance))
        else:
            # 如果死了，则重新计算生存适应度
            self.mario.calculate_fitness()
            fitness = self.mario.fitness

            # 更新最大适应度
            if fitness > self.max_fitness:
                self.max_fitness = fitness
                max_fitness = '{:.2f}'.format(self.max_fitness)
                if not args.no_display:
                    self.info_window.best_fitness.setText(max_fitness)
                    
            # 下一个体
            self._current_individual += 1

            # Are we replaying from a file?
            if args.replay_file: # 启动的是重演模式
                if not args.no_display:
                    # Set the generation to be whatever best individual is being ran (+1)
                    # Check to see if there is a next individual, otherwise exit
                    if self._current_individual >= len(args.replay_inds):
                        if args.debug:
                            print(f'Finished replaying {len(args.replay_inds)} best individuals')
                        sys.exit()
                    txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
                    self.info_window.generation.setText(txt)
                    
            else: # 启动的是演化模式
                # 是否执行下一代？
                if (self.current_generation > self._true_zero_gen and self._current_individual == self._next_gen_size) or\
                   (args.config and self.current_generation == self._true_zero_gen and self._current_individual == self._next_gen_size) or\
                    ((not args.config) and (self.current_generation == self._true_zero_gen and self._current_individual == len(args.load_inds))):
                    self.next_generation()
                # 切换当前代的下一个体
                else:
                    if self.current_generation == self._true_zero_gen:
                        if args.config:
                            # 这里其实可以并到下下一行
                            current_pop = len(self.population.individuals)
                        else:
                            current_pop = len(args.load_inds)
                    else:
                        current_pop = self._next_gen_size
                    if not args.no_display:
                        self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, current_pop))
                #print(self.mario.farthest_x,self.mario._frames,self.mario.game_score,self.mario.buttons_press_sum,'%.2f' % self.mario.fitness)
                #print(self.mario.farthest_x,'%.2f' % (-self.mario._frames*0.3),'%.2f' % (self.mario.game_score * 0.5),'%.2f' % (-self.mario.buttons_press_sum*0.1),'\n')
            if args.no_display:
                self.env.reset()  # 重新开始游戏
            else:
                self.game_window.screen = self.env.reset()  # 重新开始游戏并更新画面

            # 切换当前运行的马里奥
            self.mario = self.population.individuals[self._current_individual]
            # 切换当前绘制的马里奥神经网络
            if not args.no_display:
                self.viz.mario = self.mario
        
#程序运行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros AI')

    #配置所有运行参数
    # Config 同时也是新建模式
    parser.add_argument('-c', '--config', dest='config', required=False, help='config file to use')
    # Load arguments
    parser.add_argument('--load-file', dest='load_file', required=False, help='/path/to/population that you want to load individuals from')
    parser.add_argument('--load-inds', dest='load_inds', required=False, help='[start,stop] (inclusive) or ind1,ind2,... that you wish to load from the file')
    # No display
    parser.add_argument('--no-display', dest='no_display', required=False, default=False, action='store_true', help='If set, there will be no Qt graphics displayed and FPS is increased to max')
    # Debug
    parser.add_argument('--debug', dest='debug', required=False, default=False, action='store_true', help='If set, certain debug messages will be printed')
    # Replay arguments
    parser.add_argument('--replay-file', dest='replay_file', required=False, default=None, help='/path/to/population that you want to replay from')
    parser.add_argument('--replay-inds', dest='replay_inds', required=False, default=None, help='[start,stop] (inclusive) or ind1,ind2,ind50,... or [start,] that you wish to replay from file')
    # 手动模式(用于调试)
    parser.add_argument('--manual', dest='manual', required=False, default=False, action='store_true', help='如果设置了，则马里奥进入手动操作模式。用于调试')

    #读取输入的参数
    args = parser.parse_args()
    
    load_from_file = bool(args.load_file) and bool(args.load_inds)
    replay_from_file = bool(args.replay_file) and bool(args.replay_inds)

    # Load from file checks
    if bool(args.load_file) ^ bool(args.load_inds):
        parser.error('--load-file and --load-inds must be used together.')
    if load_from_file:
        # Convert the load_inds to be a list
        # Is it a range?
        if '[' in args.load_inds and ']' in args.load_inds:
            args.load_inds = args.load_inds.replace('[', '').replace(']', '')
            ranges = args.load_inds.split(',')
            start_idx = int(ranges[0])
            end_idx = int(ranges[1])
            args.load_inds = list(range(start_idx, end_idx + 1))
        # Otherwise it's a list of individuals to load
        elif 'last' in args.load_inds:
            file_list=[]
            for file_name in os.listdir(args.load_file+'\generations'):
                if 'npy' in file_name:
                    gen_num = int(file_name.replace('.npy','').replace('best_ind_gen',''))
                    file_list.append(gen_num)
            file_list.sort()
            if len(file_list)==0:
                parser.error(f'{args.load_file}目录下未发现存档。如果需要新建，请使用-c命令新建')
            elif len(file_list)==1:
                args.load_inds = file_list[-1:]
            else:
                args.load_inds = file_list[-2:]
        else:
            args.load_inds = [int(ind) for ind in args.load_inds.split(',')]

    # Replay from file checks
    if bool(args.replay_file) ^ bool(args.replay_inds):
        parser.error('--replay-file and --replay-inds must be used together.')
    if replay_from_file:
        # Convert the replay_inds to be a list
        # is it a range?
        if '[' in args.replay_inds and ']' in args.replay_inds:
            args.replay_inds = args.replay_inds.replace('[', '').replace(']', '')
            ranges = args.replay_inds.split(',')
            has_end_idx = bool(ranges[1])
            start_idx = int(ranges[0])
            # Is there an end idx? i.e. [12,15]
            if has_end_idx:
                end_idx = int(ranges[1])
                args.replay_inds = list(range(start_idx, end_idx + 1))
            # Or is it just a start? i.e. [12,]
            else:
                end_idx = start_idx
                for fname in os.listdir(args.replay_file):
                    if fname.startswith('best_ind_gen'):
                        ind_num = int(fname[len('best_ind_gen'):])
                        if ind_num > end_idx:
                            end_idx = ind_num
                args.replay_inds = list(range(start_idx, end_idx + 1))
        # Otherwise it's a list of individuals
        else:
            args.replay_inds = [int(ind) for ind in args.replay_inds.split(',')]

    if replay_from_file and load_from_file:
        parser.error('Cannot replay and load from a file.')

    # Make sure config AND/OR [(load_file and load_inds) or (replay_file and replay_inds)]
    # 新建模式，载入模式，重演模式，你必须选择一个
    if not (bool(args.config) or (load_from_file or replay_from_file)):
        parser.error('Must specify -c and/or [(--load-file and --load-inds) or (--replay-file and --replay-inds)]')
    return args

if __name__ == "__main__":
    global args
    # 加载运行参数
    args = parse_args()
    config = None
    # 加载配置文件
    if args.config:
        config = Config(args.config)
    else:
        config = Config(args.load_file)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())
