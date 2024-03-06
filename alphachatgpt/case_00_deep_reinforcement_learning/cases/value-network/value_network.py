import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义棋盘大小
BOARD_SIZE = 15

# 定义五子棋游戏状态类
class GameState:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

    def get_legal_moves(self):
        # 获取所有合法的落子位置
        return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if self.board[i][j] == 0]

    def make_move(self, move, player):
        # 在指定位置落子
        self.board[move[0]][move[1]] = player

    def check_win(self, player):
        # 检查是否获胜
        # 此处省略五子连珠的判断逻辑
        pass

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 64 * BOARD_SIZE * BOARD_SIZE)
        x = torch.softmax(self.fc(x), dim=1)
        return x

# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 64 * BOARD_SIZE * BOARD_SIZE)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# 计算价值网络的熵
def calculate_value_entropy(value_network, game_state):
    board_state = torch.FloatTensor(game_state.board).unsqueeze(0).unsqueeze(0)
    value = value_network(board_state)
    prob = torch.softmax(value, dim=1)
    log_prob = torch.log(prob + 1e-6)
    entropy = -torch.sum(prob * log_prob)
    return entropy.item()

# 示例代码
policy_net = PolicyNetwork()
value_net = ValueNetwork()

game_state = GameState()
move = (7, 7)  # 例如，落子在中心位置
player = 1  # 假设当前玩家为1

game_state.make_move(move, player)

# 计算价值网络的熵
value_entropy = calculate_value_entropy(value_net, game_state)
print("Value Network Entropy:", value_entropy)
