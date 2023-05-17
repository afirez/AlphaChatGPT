
from surprise import Dataset, KNNBasic
from surprise.model_selection import train_test_split

"""
Prompt: 分别基于 KNN, Surprise，LightFM， Tensorflow， Pytorch 实现推荐系统

Surprise 部分
"""



# 加载数据集
data = Dataset.load_builtin('ml-100k')

# 划分训练集和测试集
trainset, testset = train_test_split(data, test_size=0.2)

# 构建协同过滤模型
model = KNNBasic()

# 训练模型
model.fit(trainset)

# 预测评分
predictions = model.test(testset)

# 输出前几个预测结果
for uid, iid, true_r, est, _ in predictions[:5]:
    print(f"User: {uid}, Item: {iid}, True rating: {true_r}, Predicted rating: {est}")
