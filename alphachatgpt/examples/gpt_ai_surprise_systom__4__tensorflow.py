
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.optimizers import Adam

"""
Prompt: 分别基于 KNN, Surprise，LightFM， Tensorflow， Pytorch 实现推荐系统

Tensorflow 部分
"""

# 构建模型
num_users = 1000
num_items = 500
embedding_size = 32

user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(num_users, embedding_size)(user_input)
item_embedding = Embedding(num_items, embedding_size)(item_input)

user_flat = Flatten()(user_embedding)
item_flat = Flatten()(item_embedding)

concat = tf.concat([user_flat, item_flat], axis=1)
output = Dense(1, activation='sigmoid')(concat)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# 训练模型
model.fit([user_ids, item_ids], labels, epochs=10, batch_size=64)

# 生成推荐结果
predictions = model.predict([user_ids, item_ids])

# 输出前几个预测结果
for i in range(5):
    print(f"User: {user_ids[i]}, Item: {item_ids[i]}, Predicted rating: {predictions[i][0]}")

