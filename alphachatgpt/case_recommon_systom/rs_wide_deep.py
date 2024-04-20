import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义Wide & Deep模型
class WideAndDeep(nn.Module):
    def __init__(self, num_wide_features, embedding_dim, hidden_dim):
        super(WideAndDeep, self).__init__()
        
        # Wide部分，直接连接到输出
        self.wide_linear = nn.Linear(num_wide_features, 1)
        
        # Deep部分，使用Embedding层和多层全连接层
        self.embedding = nn.Embedding(num_embeddings=num_wide_features, embedding_dim=embedding_dim)
        self.deep_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.deep_fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, wide_input, deep_input):
        # Wide部分
        wide_output = self.wide_linear(wide_input)
        
        # Deep部分
        deep_embed = self.embedding(deep_input)
        deep_embed = deep_embed.mean(dim=1)  # 池化操作，使用均值池化
        deep_output = F.relu(self.deep_fc1(deep_embed))
        deep_output = self.deep_fc2(deep_output)
        
        # Wide和Deep部分的输出相加
        output = wide_output + deep_output
        return output

# 示例数据
wide_features = torch.randn(100, 10)  # Wide特征，假设有10个特征
deep_features = torch.randint(0, 100, (100, 5))  # Deep特征，假设有5个类别特征，每个特征取值范围是[0, 99]
labels = torch.randn(100, 1)

# 创建数据加载器
dataset = TensorDataset(wide_features, deep_features, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
model = WideAndDeep(num_wide_features=10, embedding_dim=10, hidden_dim=50)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for wide_input, deep_input, label in loader:
        optimizer.zero_grad()
        
        output = model(wide_input, deep_input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")

# 在实际应用中，可能还需要对模型进行评估和调参等步骤