import torch
import torch.nn as nn
from math import sqrt

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 256):
        super().__init__()

        # pos = 0, 1, …, max_len-1
        pos = torch.arange(max_len).unsqueeze(1)
        # 2i = 0, 2, …, emb_dim-2
        two_i = torch.arange(0, emb_dim, 2)
        # pos / 10000^(2i/d)   
        angle = pos / 10000 ** (two_i / emb_dim)

        pe = torch.zeros(max_len, emb_dim)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        all_heads_dim = n_heads * head_dim

        self.query = nn.Linear(emb_dim, all_heads_dim)
        self.key = nn.Linear(emb_dim, all_heads_dim)
        self.value = nn.Linear(emb_dim, all_heads_dim)
        self.output = nn.Linear(all_heads_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.n_heads * self.head_dim)

    def forward(self, x, attention_mask=None):
        Q = self.split_heads(self.query(x))
        K = self.split_heads(self.key(x))
        V = self.split_heads(self.value(x))

        scores = Q @ K.transpose(-2, -1) / sqrt(self.head_dim)

        # Padding tokens get -inf, so they receive zero attention after the softmax
        if attention_mask is not None:
            is_padding = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(is_padding, float("-inf"))

        # Turn scores into weights that sum to 1 for each token
        attention_weights = scores.softmax(dim=-1)

        out = self.dropout(attention_weights) @ V

        # Put the heads back together and mix them
        out = self.output(self.merge_heads(out))
        return out, attention_weights


class PositionwiseFFN(nn.Module):
    def __init__(self, emb_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, head_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadSelfAttention(emb_dim, n_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.pos_wise_ffn = PositionwiseFFN(emb_dim, feed_forward_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        attn_out, attn_weights = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.pos_wise_ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout_p = 0.1
        self.setup_model()

    def setup_model(self):
        # Embedding(number of dim (30522/bert vocab size), vector length, padding id(0 standard for bert))
        self.token_embedding = nn.Embedding(30522, self.emb_dim, 0)
        self.positional_encoding = PositionalEncoding(self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        layers = []
        for _ in range(self.n_layers):
            layers.append(TransformerLayer(self.emb_dim, self.n_heads, self.hidden_dim, 
                                           self.feed_forward_dim, self.dropout_p))
        
        self.layers = nn.ModuleList(layers)

    def pool(self, x):
        if self.cfg.model.pooling_type == 'cls':
            return x[:, 0, :]
        
        elif self.cfg.model.pooling_type == 'max':
            return x.max(dim=1).values
        
        elif self.cfg.model.pooling_type == 'mean':
            return x.mean(dim=1)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.token_embedding(x) * sqrt(self.emb_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.pool(x)               