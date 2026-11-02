import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query 

        Returns: A 1d torch tensor 
        """
        # (i, j) = inner product between query i and document j -> batch_size x batch_size
        similarities = queries @ documents.T

        # Document i is the correct answer for query i (the diagonal). The other documents in the batch are wrong answers.
        targets = torch.arange(queries.size(0), device=queries.device)

        return nn.functional.cross_entropy(similarities, targets)
