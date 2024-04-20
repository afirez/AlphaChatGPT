import pandas as pd
import matplotlib.pyplot as plt


# 画两个图：loss和acc分开
# 可视化train_loss
data = pd.read_csv('one_way_lstm.csv')
x = range(0, 1950)
y = data['train_loss']
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', label=u'train_loss')
plt.ylim([0, 1])
plt.title(u'train_loss and test_acc')
plt.legend()
plt.ylabel('loss')

# 可视化测试集准确率
test_data = pd.read_csv('test_acc_one_way_epoch8.csv')
x = range(0, 50)
y = test_data['test_acc']
plt.subplot(2, 1, 2)
plt.plot(x, y, 'r-', label=u'test_acc')
plt.legend()
plt.xlabel('iterations & epochs')
plt.ylabel('acc')
plt.ylim([0, 1])

plt.show()


