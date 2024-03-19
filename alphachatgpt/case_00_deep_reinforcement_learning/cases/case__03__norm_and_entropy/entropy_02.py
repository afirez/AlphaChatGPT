import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
用 pytorch 实现代码，方便理解模型输出的熵，激活函数的熵，梯度的熵，误差的熵，参数的熵
"""

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 计算模型输出的熵
def compute_output_entropy(outputs):
    entropy = -torch.sum(outputs * torch.log(outputs), dim=1)
    return entropy.mean().item()

# 计算激活函数的熵
def compute_activation_entropy(activations):
    entropy = -torch.sum(activations * torch.log(activations), dim=1)
    return entropy.mean().item()

# 计算梯度的熵
def compute_gradient_entropy(gradients):
    flat_gradients = gradients.view(-1)
    entropy = -torch.sum(flat_gradients * torch.log(flat_gradients))
    return entropy.item()

# 计算误差的熵
def compute_error_entropy(predictions, targets):
    entropy = nn.CrossEntropyLoss()(predictions, targets)
    return entropy.item()

# 计算参数的熵
def compute_parameters_entropy(model):
    flat_params = torch.cat([param.view(-1) for param in model.parameters()])
    entropy = -torch.sum(flat_params * torch.log(flat_params))
    return entropy.item()

# 生成示例数据
np.random.seed(42)
torch.manual_seed(42)
inputs = torch.randn(100, 2)
targets = torch.randint(0, 2, (100,))

# 实例化模型和优化器
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前向传播
outputs = model(inputs)
predictions = torch.argmax(outputs, dim=1)

# 反向传播计算梯度
optimizer.zero_grad()
loss = nn.CrossEntropyLoss()(outputs, targets)
loss.backward()

# 获取激活函数的输出
activations = model.relu(model.fc1(inputs))

# 获取梯度
gradients = [param.grad for param in model.parameters()]

# 计算模型输出的熵
output_entropy = compute_output_entropy(outputs)
print("Output entropy:", output_entropy)

# 计算激活函数的熵
activation_entropy = compute_activation_entropy(activations)
print("Activation entropy:", activation_entropy)

# 计算梯度的熵
gradient_entropy = compute_gradient_entropy(gradients)
print("Gradient entropy:", gradient_entropy)

# 计算误差的熵
error_entropy = compute_error_entropy(outputs, targets)
print("Error entropy:", error_entropy)

# 计算参数的熵
parameters_entropy = compute_parameters_entropy(model)
print("Parameters entropy:", parameters_entropy)
