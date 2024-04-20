## 推荐系统 Wide & Deep

### ChatGPT QA
User
（1）排序模型一般都衍生自Google的Wide & Deep模型，有一个浅层模型（LR或FM）负责记忆，DNN负责扩展

（2）特征一般都采用类别特征。画像、用户历史天然就是高维、稀疏的类别特征。对于实数型特征，比如用户、物料的一些统计指标，在我的实践中，也通过分桶，先离散化成类别特征，再接入模型

（3）每个类别特征经过Embedding变成一个向量，以扩展其内涵。

（4）属于一个Field的各Feature Embedding需要通过Pooling压缩成一个向量，以减少DNN的规模

（5）多个Field Embedding拼接在一起，喂入DNN

（6）DNN通过多层Fully Connection Layer (FC)完成特征之间的高阶交叉，增强模型的扩展能力。

ChatGPT
这些观点涉及到排序模型的基本结构和特征处理方法。让我总结一下：

Wide & Deep 模型结构: Wide & Deep模型结构由两部分组成：Wide部分是浅层模型（如逻辑回归LR或因子分解机FM），用于记忆特征之间的线性关系；Deep部分是深度神经网络（DNN），用于学习特征之间的高阶交互关系。
特征处理:
类别特征: 通常用于描述画像、用户历史等信息，这些特征通常是高维稀疏的，可以直接输入模型进行处理。
实数型特征: 例如统计指标，可以通过分桶转化成类别特征，然后输入模型。
Embedding: 每个类别特征通过Embedding层转化为向量，以便在模型中进行处理，从而扩展特征的内涵。
Pooling: 每个Field的各Feature Embedding经过Pooling压缩成一个向量，以减少DNN的规模，通常使用池化操作如平均池化或最大池化。
特征拼接: 多个Field的Embedding拼接在一起，作为DNN的输入。
DNN结构: DNN通常由多层Fully Connection Layer (FC)组成，用于学习特征之间的高阶交叉，增强模型的扩展能力。
这些方法在排序模型中被广泛使用，帮助模型更好地处理不同类型的特征，提高排序性能。

User
给出pytorch 代码实现，以便更好的理解 

ChatGPT
当然，我可以提供一个简单的示例代码，演示如何使用PyTorch实现Wide & Deep模型。以下是一个简化版本的代码示例：

```python
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
```

这是一个简单的示例，实现了Wide & Deep模型的基本结构。在实际应用中，你可能需要根据具体任务和数据的特点进行更复杂的模型设计和调优。