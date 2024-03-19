import torch

# 定义马尔可夫链的状态空间和转移概率矩阵
states = ['Sunny', 'Rainy']
transition_probabilities = torch.tensor([[0.8, 0.2],  # Sunny -> Sunny, Sunny -> Rainy
                                         [0.4, 0.6]]) # Rainy -> Sunny, Rainy -> Rainy

# 生成观测数据
observations = ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny']

# 初始化转移概率矩阵的估计
estimated_transition_probabilities = torch.rand(2, 2, requires_grad=True)

# 定义优化器
optimizer = torch.optim.Adam([estimated_transition_probabilities], lr=0.1)

# 进行估计
num_epochs = 1000
for epoch in range(num_epochs):
    # 计算似然函数
    log_likelihood = 0
    for i in range(len(observations) - 1):
        current_state_index = states.index(observations[i])
        next_state_index = states.index(observations[i + 1])
        log_likelihood += torch.log(estimated_transition_probabilities[current_state_index, next_state_index])
    
    # 最大化似然函数等价于最小化负的似然函数
    loss = -log_likelihood
    
    # 梯度清零
    optimizer.zero_grad()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    # 打印当前的损失值
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 输出估计的转移概率矩阵
print("Estimated transition probabilities:")
print(estimated_transition_probabilities.detach().numpy())
