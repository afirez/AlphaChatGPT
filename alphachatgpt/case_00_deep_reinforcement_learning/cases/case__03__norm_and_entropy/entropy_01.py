import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义一个简单的分类模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))

# 训练函数（应用L2范数正则化）
def train_with_regularization(model, criterion, optimizer, num_epochs, reg_lambda):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 计算L2范数正则化项
        l2_regularization = torch.tensor(0.)
        for param in model.parameters():
            l2_regularization += torch.norm(param, 2)

        # 添加正则化项到损失函数中
        loss += reg_lambda * l2_regularization

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')



# 生成示例数据
np.random.seed(42)
torch.manual_seed(42)
input_dim = 2
output_dim = 2
num_samples = 100
inputs = torch.randn(num_samples, input_dim)
targets = torch.randint(0, output_dim, (num_samples,))

# 定义模型、损失函数和优化器
model = SimpleClassifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型（应用L2范数正则化）
num_epochs = 500
reg_lambda = 0.01  # 正则化系数
train_with_regularization(model, criterion, optimizer, num_epochs, reg_lambda)


# 定义损失函数为交叉熵
criterion = nn.CrossEntropyLoss()

# 最大化模型信息论熵
def maximize_entropy(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = -torch.mean(torch.sum(outputs * torch.log(outputs), dim=1))  # 最大化熵等价于最小化负熵

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 最小化模型信息论熵
def minimize_entropy(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = torch.mean(torch.sum(outputs * torch.log(outputs), dim=1))  # 最小化熵

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

def train_2():
    # 定义一个新的分类模型
    model = SimpleClassifier(input_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型（最大化熵）
    maximize_entropy(model, criterion, optimizer, num_epochs)

    # 重新初始化模型和优化器
    model = SimpleClassifier(input_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型（最小化熵）
    minimize_entropy(model, criterion, optimizer, num_epochs)
