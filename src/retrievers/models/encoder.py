import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        # Embedding(number of dim (30522/bert vocab size), vector length, padding id(0 standard for bert))
        self.token_embedding = nn.Embedding(30522, self.emb_dim, 0)
        self.positional_encoding = NotImplementedError()
        self.dropout = NotImplementedError()
        self.layers = NotImplementedError()
        self.final_norm = NotImplementedError()

        raise NotImplementedError()



    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            return x[:, 0, :]
        
        elif self.cfg.model.pooling_type == 'max':
            return x.max(dim=1).values
        
        elif self.cfg.model.pooling_type == 'mean':
            return x.mean(dim=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        self.token_embedding(x) * sqrt(self.emb_dim)

        #self.positional_encoding(x)
        #self.dropout(x)
        #self.layers(x)
        #self.final_norm(x)
        return 