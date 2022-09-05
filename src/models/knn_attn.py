import torch
import torch.nn as nn
import math


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, key, query, mask=None):
        """_summary_
        Args:
            key (_type_): tensor([1,n_station,d_dim])
            query (_type_): tensor([1,n_station,d_dim])
            mask (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        n_dim = key.shape[-1]
        n_station = key.shape[1]
        query = query.unsqueeze(1)
        score = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(
            torch.tensor([n_dim], device="cuda")
        )
        if mask is not None:
            mask = mask.squeeze()
            score = score.masked_fill(mask == 0, -math.inf)
        attn = self.softmax(score.view(-1, n_station))
        return attn


