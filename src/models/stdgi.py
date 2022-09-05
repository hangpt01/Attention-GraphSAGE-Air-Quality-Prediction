from src.models.graphsage import DotProductAttention, GraphSage
from src.models.discriminator import Discriminator
import numpy as np
import torch.nn as nn


class STDGI(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        dis_hid,
        aggregator,
        k,
        stdgi_noise_min=0.4,
        stdgi_noise_max=0.7,
    ):
        super(STDGI, self).__init__()
        print("Init Attention_Encoder model ...")
        # self.encoder = TGCN_Encoder(
        #         in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
        #     )

        self.encoder = GraphSage(aggregator, k, in_ft, out_ft)
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max

    def forward(self, x, x_k, adj):
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        ret = self.disc(h, x_k, x_c)
        return ret

    def corrupt(self, X):
        """
        X: [batch_size,n_stas,n_fts]
        """
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts


import torch

if __name__ == "__main__":
    aggr = DotProductAttention()
    stdgi = STDGI(128, 128, 6, aggr, 3)
    x = torch.rand(12, 6, 128)
    g = torch.rand(12, 6, 6)
    l = torch.rand(12, 6)
    h = stdgi(x, x, g)
    k = stdgi.encoder.inductive(x,l,g)
    breakpoint()