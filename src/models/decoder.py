from src.models.graphsage import DotProductAttention
import torch.nn as nn
import torch as torch



class Decoder(nn.Module):
    def __init__(self,hid_ft,n_rnns,):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_size=hid_ft,hidden_size=hid_ft,batch_first=True, num_layers=n_rnns)
        self.linear = nn.Linear(hid_ft,1)
        self.activation = nn.ReLU()
        self.attn = DotProductAttention()
        self.w_key = nn.Linear(hid_ft,hid_ft)
        self.w_query = nn.Linear(hid_ft,hid_ft)
        self.w_value = nn.Linear(hid_ft,hid_ft)
        self.w_k = nn.Linear(2*hid_ft,hid_ft)
    def forward(self,H,l):
        knn_station = torch.argsort(l.squeeze(), -1)[:, -3:]
        mask = torch.zeros_like(l, dtype=torch.int32)
        list_h = []

        for j in range(l.shape[0]):
            for i in knn_station[j]:
                mask[j, i] = 1

        for _ in range(H.shape[1]):
            h = H[:,_,:,:]
            idw_vector = torch.bmm(l.unsqueeze(1),h)
            h_x = idw_vector
            h_x_attn_score = self.attn(self.w_key(h),self.w_query(h_x),mask.unsqueeze(1))
            h_kn_x = torch.bmm(h_x_attn_score,self.w_value(h))
            h_x = self.activation(self.w_k(torch.concat((h_x,h_kn_x),dim=-1)))
            list_h.append(h_x.squeeze())
        h = torch.stack(list_h,1)
        # breakpoint()
        h_, _ = self.gru(h)
        return self.activation(self.linear(h_[:,-1,:]))

