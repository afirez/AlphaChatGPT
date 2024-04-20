import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import torch.nn.functional as F

import math


"""
QKV 注意力
全连接自适应相似性计算
"""

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, attention_dim: int):
        super(SingleHeadAttention, self).__init__()
        
        self.query_proj = nn.Linear(embed_dim, attention_dim)
        self.key_proj = nn.Linear(embed_dim, attention_dim)
        self.value_proj = nn.Linear(embed_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, embed_dim)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.shape[-1])
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        return self.out_proj(attn_output)


class AT_BiLSTM(nn.Module):
    def __init__(
        self, 
        vocab: nn.Embedding, 
        batch_size: int = 128, 
        output_size: int = 2, 
        hidden_size: int = 256,
        embed_dim: int = 300, 
        dropout: float = 0.1, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        attention_dim: int = 256, 
        sequence_length: int = 512, 
        bidirectional: bool = True
    ):
        super(AT_BiLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.device = device
        self.sequence_length = sequence_length

        self.embedding = nn.Embedding.from_pretrained(vocab.get_vecs_by_tokens(vocab.get_itos()), 
                                                     padding_idx=vocab['<pad>'], 
                                                     freeze=True)

        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=bidirectional, dropout=dropout)
        self.layer_size = 2 if bidirectional else 1
        
        self.attention = SingleHeadAttention(hidden_size * self.layer_size, attention_dim)
        
        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    def forward(self, input_sentences: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        input = self.embedding(input_sentences).permute(1, 0, 2).to(self.device)

        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)

        lstm_output, _ = self.lstm(input, (h_0, c_0))
        attn_output = self.attention(lstm_output)
        logits = self.label(attn_output)
        return logits