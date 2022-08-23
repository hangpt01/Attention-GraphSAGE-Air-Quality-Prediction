from src.models.discriminator import Discriminator
from src.models.encoder import TGCN_Encoder
import numpy as np 
import torch.nn as nn


class STDGI(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        en_hid1,
        en_hid2,
        dis_hid,
        act_en="relu",
        stdgi_noise_min=0.4,
        stdgi_noise_max=0.7,
        model_type="gede",
        num_input_station=0,
    ):
        super(STDGI, self).__init__()
        print("Init Attention_Encoder model ...")
        self.encoder = TGCN_Encoder(
                in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
            )

        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max

    def forward(self, x, x_k, adj):
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        ret = self.disc(h[:, -1, :, :], x_k[:, -1, :, :], x_c[:, -1, :, :])
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts

    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h
