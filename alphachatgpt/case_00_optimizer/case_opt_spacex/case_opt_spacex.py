import torch
import torch.optim as optim

# 定义火箭的初始状态和参数
initial_position = torch.tensor([0.0, 0.0], requires_grad=True)
initial_velocity = torch.tensor([50.0, 100.0], requires_grad=True)
gravity = torch.tensor([0.0, -9.8])  # 重力加速度
air_resistance = 0.01  # 大气阻力系数

# 定义优化器和学习率
optimizer = optim.SGD([initial_position, initial_velocity], lr=0.01)

# 定义着陆目标点
landing_target = torch.tensor([100.0, 0.0])

# 定义优化迭代次数
num_iterations = 1000

# 进行优化迭代
for i in range(num_iterations):
    # 计算当前位置到目标点的距离
    distance_to_target = torch.norm(initial_position - landing_target)

    # 计算当前速度的方向
    velocity_direction = initial_velocity / torch.norm(initial_velocity)

    # 计算当前速度受到的空气阻力
    air_resistance_force = -air_resistance * torch.norm(initial_velocity) * velocity_direction

    # 计算当前受到的总加速度
    total_acceleration = gravity + air_resistance_force

    # 更新速度和位置
    initial_velocity = initial_velocity + total_acceleration
    initial_position = initial_position + initial_velocity

    # 计算损失函数，即当前位置到目标点的距离
    loss = distance_to_target

    # 清除梯度
    optimizer.zero_grad()

    # 计算损失函数关于初始位置和速度的梯度
    loss.backward()

    # 使用梯度下降更新参数
    optimizer.step()

    # 打印优化过程中的损失值
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}")

# 打印最终优化结果
print("Final optimized position:", initial_position.detach().numpy())