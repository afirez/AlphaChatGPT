
import torch
import torch.nn as nn
import torch.optim as optim

"""
Prompt: 分别基于 KNN, Surprise，LightFM， Tensorflow， Pytorch 实现推荐系统

Pytorch
"""

# 构建模型
num_users = 1000
num_items = 500
embedding_size = 32

class Recommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(embedding_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        user_flat = self.flatten(user_embedded)
        item_flat = self.flatten(item_embedded)
        concat = torch.cat((user_flat, item_flat), dim=1)
        output = self.dense(concat)
        output = self.sigmoid(output)
        return output

# 创建模型实例
model = Recommender(num_users, num_items, embedding_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转换数据为Tensor
user_ids = torch.tensor(user_ids, dtype=torch.long)
item_ids = torch.tensor(item_ids, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.float)

# 训练模型
num_epochs = 10
batch_size = 64
num_samples = len(user_ids)

for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        user_batch = user_ids[i:i+batch_size]
        item_batch = item_ids[i:i+batch_size]
        label_batch = labels[i:i+batch_size]

        optimizer.zero_grad()
        output = model(user_batch, item_batch)
        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()

# 生成推荐结果
predictions = model(user_ids, item_ids)

# 输出前几个预测结果
for i in range(5):
    print(f"User: {user_ids[i]}, Item: {item_ids[i]}, Predicted rating: {predictions[i].item()}")


