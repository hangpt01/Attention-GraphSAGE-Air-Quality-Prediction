import torch.nn as nn
import torch as torch



class Decoder(nn.Module):
    def __init__(self,hid_ft,n_rnns,):
        super(Decoder, self).__init__()
        self.gru = nn.LSTM(input_size=hid_ft,hidden_size=hid_ft,batch_first=True, num_layers=n_rnns)
        self.linear = nn.Linear(hid_ft,1)
        self.activation = nn.ReLU()

    def forward(self,h):
        h_, _ = self.gru(h)
        return self.activation(self.linear(h_[:,-1,:]))

