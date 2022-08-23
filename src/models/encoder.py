import torch.nn as nn 
from src.models.temporal_gcn import TemporalGCN
import torch

class TGCN_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(TGCN_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn_gcn = TemporalGCN(hid_ft1, out_ft, hid_ft2)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x)) # 12, 19, 200 
        raw_shape = x.shape
        h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device=torch.device('cuda')) #(1, 19, 400)
        # breakpoint()
        list_h = []
        for i in range(raw_shape[1]):
            x_i = x[:,i,:,:].squeeze(1) # 1, 19, 200 
            e = adj[:,i,:,:].squeeze(1) # 1, 19, 19
            h = self.rnn_gcn(x_i, e, h)
            list_h.append(h)
        h_ = torch.stack(list_h, dim=1)
        return h_