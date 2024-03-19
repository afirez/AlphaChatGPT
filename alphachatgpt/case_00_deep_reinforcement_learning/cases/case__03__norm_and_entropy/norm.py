import torch
import torch.nn as nn
import torch.optim as optim

"""
解空间：
  解空间指的是在某个问题的上下文中，可能的解决方案所构成的空间。
  在机器学习中，解空间通常是指模型参数的可能取值范围。
  例如，在线性回归问题中，解空间是所有可能的权重参数的集合。

模型权重参数 正则化范数： 
  1. L1 范数
  2. L2 范数

范数收敛：
范数收敛指的是在某种范数下的收敛性质。
在优化问题中，通常会关注参数的收敛性，而范数是衡量参数变化的一种方法。
如果优化算法在某种范数下的参数变化趋于零，那么可以说该算法在该范数下达到了收敛状态。
范数收敛是许多优化算法（如梯度下降法）收敛性分析的重要指标之一。

"""

# 定义一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 训练函数
def train(model, criterion, optimizer, num_epochs, reg_lambda=None, reg_type=None):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 添加正则化项
        if reg_lambda and reg_type:
            if reg_type == 'L1':
                l1_regularization = torch.norm(model.linear.weight, 1)
                loss += reg_lambda * l1_regularization
            elif reg_type == 'L2':
                l2_regularization = torch.norm(model.linear.weight, 2)
                loss += reg_lambda * l2_regularization

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 输入数据和目标
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 模型、损失函数和优化器
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 500
reg_lambda = 0.01  # 正则化系数
reg_type = 'L2'  # 正则化类型（可以是'L1'或'L2'）
train(model, criterion, optimizer, num_epochs, reg_lambda, reg_type)
