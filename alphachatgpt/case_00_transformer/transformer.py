import torch.nn as nn
import torch

"""
Transformer Attention q,k,v
"""

class TransformerBlock(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        assert x.shape[1] == self.input_dim, "输入矩阵的列数需要和模型的维度一致"

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 计算注意力分数并进行归一化
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.input_dim))

        # 进行 softmax 操作获取权重分布
        attn_weights = torch.softmax(attention_scores, dim=-1)

        # 使用权重分布加权求和得到输出值
        out = torch.matmul(attn_weights, v)

        return out
    
    def update_weight(self):
        self.w_q.weight.data -= self.w_q.weight.grad
        self.w_q.bias.data -= self.w_q.bias.grad

        # 清零梯度
        self.w_q.weight.grad.zero_()
        self.w_q.bias.grad.zero_()

def case_1():
    input = torch.randn(8, 8) # 假设我们有一个 382 x 382 的输入矩阵
    print(input)

    model = TransformerBlock(input_dim=8)
    output = model(input)
    print(f"w_q.bias:{model.w_q.bias}")
    print(output)

    loss = 3 - torch.sum(output)
    loss.backward()

    model.update_weight()

    torch.save(model.state_dict(), "model.pth")

    model = TransformerBlock(input_dim=8)
    model.load_state_dict(torch.load("model.pth"))
    print(f"w_q.bias:{model.w_q.bias}")

case_1()