# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch

@author: Xiang Zhong
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        
        # 公共网络层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 动作网络
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                board_width * board_height)
        
        # 评估网络
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 公共层
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 动作网络
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)
        
        # 评估网络
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        
        return x_act, x_val


class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2正则化系数
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), 
                                  weight_decay=self.l2_const)
        
        if model_file is not None:
            self.restore_model(model_file)
            
    def policy_value(self, state_batch):
        """
        输入: state批量数据
        输出: 动作概率和状态价值
        """
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs, value.cpu().numpy()
            
    def policy_value_fn(self, board):
        """
        输入: 棋盘
        输出: 每个可用动作的(action, probability)元组列表和棋盘状态的分数
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value[0][0]
        
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一步训练"""
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)
        
        self.policy_value_net.train()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.view(-1, 1)
        
        # 计算损失
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        
        # 反向传播和优化
        loss.backward()
        self.optimizer.step()
        
        # 计算entropy，仅用于监控
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        ).item()
        
        return loss.item(), entropy
        
    def save_model(self, model_path):
        """保存模型"""
        torch.save(self.policy_value_net.state_dict(), model_path)
        
    def restore_model(self, model_path):
        """加载模型"""
        self.policy_value_net.load_state_dict(torch.load(model_path))