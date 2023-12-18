import numpy as np
import matplotlib.pyplot as plt

# 定义波函数
def wave_function(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma)**2) * np.exp(1j * k * x)

# 参数设置
x = np.linspace(-10, 10, 1000)  # 空间范围
k = 1  # 波数
mean = 0  # 均值
sigma = 1  # 标准差

# 计算波函数
psi = wave_function(x, mean, sigma)

# 计算概率波
probability_density = np.abs(psi)**2

# 绘制波函数和概率波
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, psi.real, label='Real part')
plt.plot(x, psi.imag, label='Imaginary part')
plt.title('Wave Function')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, probability_density, color='green', label='Probability Density')
plt.title('Probability Wave (Probability Density)')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.legend()

plt.tight_layout()
plt.show()
