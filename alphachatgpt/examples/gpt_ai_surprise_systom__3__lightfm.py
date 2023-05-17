
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

"""
Prompt: 分别基于 KNN, Surprise，LightFM， Tensorflow， Pytorch 实现推荐系统

LightFM 部分
"""

# 加载Movielens数据集
data = fetch_movielens()

# 构建模型
model = LightFM(loss='warp')  # 使用WARP损失函数

# 训练模型
model.fit(data['train'], epochs=10, num_threads=2)

# 评估模型
precision = precision_at_k(model, data['test'], k=5).mean()

print(f"Precision at k: {precision}")

