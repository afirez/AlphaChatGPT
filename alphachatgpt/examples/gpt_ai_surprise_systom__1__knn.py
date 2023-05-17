
import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
Prompt: 分别基于 KNN, Surprise，LightFM， Tensorflow， Pytorch 实现推荐系统

KNN
"""


# 创建用户-物品矩阵
user_item_matrix = np.array([
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 1]
])

# 创建物品-物品相似度矩阵
item_similarity_matrix = np.dot(user_item_matrix.T, user_item_matrix)

# 创建NearestNeighbors模型
k = 2  # 设置邻居数量
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(item_similarity_matrix)

# 根据用户的喜好推荐物品
user_id = 0  # 用户ID
user_vector = user_item_matrix[user_id]
_, indices = model.kneighbors([user_vector], n_neighbors=k)

# 输出推荐的物品
recommended_items = indices.flatten()
print("Recommended items for user", user_id)
for item_id in recommended_items:
    if user_vector[item_id] == 0:  # 排除用户已经喜欢的物品
        print("Item", item_id)
