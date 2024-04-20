import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, input_dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, query, key, value):
        # Linear transformation
        query_linear = self.linear(query)
        key_linear = self.linear(key)
        
        # Attention scores
        attention_scores = torch.matmul(query_linear, key_linear.transpose(-2, -1))

        use_softmax = False # 线性注意力中 softmax 操作是必要的吗
        if use_softmax:
            # Attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)
        else:
            attention_weights = attention_scores

        # Weighted sum
        output = torch.matmul(attention_weights, value)
        return output

class FastTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(FastTransformerLayer, self).__init__()
        self.attention = LinearAttention(input_dim)
        self.feed_forward = nn.Linear(input_dim, input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.num_heads = num_heads
    
    def forward(self, x):
        # Self-attention
        residual = x
        x_normalized = self.layer_norm1(x)
        query = x_normalized
        key = x_normalized
        value = x_normalized
        attention_output = self.attention(query, key, value)
        x = residual + attention_output
        
        # Feed-forward network
        residual = x
        x_normalized = self.layer_norm2(x)
        x = F.relu(self.feed_forward(x_normalized))
        x = residual + x
        return x

class FastTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=6):
        super(FastTransformer, self).__init__()
        self.layers = nn.ModuleList([FastTransformerLayer(input_dim) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
input_dim = 512
sequence_length = 10
batch_size = 32

# Create dummy input
input_data = torch.randn(batch_size, sequence_length, input_dim)

# Initialize Fast Transformer model
fast_transformer = FastTransformer(input_dim=input_dim)

# Forward pass
output = fast_transformer(input_data)
print(output.shape)  # Output shape: (batch_size, sequence_length, input_dim)