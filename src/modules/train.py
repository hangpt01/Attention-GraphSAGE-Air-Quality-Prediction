import torch
from tqdm import tqdm


def train_stdgi(stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2):
    # wandb.watch(stdgi, criterion, log="all", log_freq=100)
    """
    Sử dụng train Attention_STDGI model
    """
    epoch_loss = 0
    stdgi.train()
    for data in tqdm(dataloader):
        for i in range(n_steps):
            optim_d.zero_grad()
            d_loss = 0
            x = data["X"][:,-1,:,:].to(device).float()
            # breakpoint()
            G = data["G2"].to(device).float()
            output = stdgi(x, x, G)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            d_loss = criterion(output, lbl)
            d_loss.backward()
            optim_d.step()

        optim_e.zero_grad()
        x = data["X"][:,-1,:,:].to(device).float()
        G = data["G2"].to(device).float()
        output = stdgi(x, x, G)
        lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
        lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
        lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
        e_loss = criterion(output, lbl)
        e_loss.backward()
        optim_e.step()
        epoch_loss += e_loss.detach().cpu().item()
    return epoch_loss / len(dataloader)
import wandb

def train_atten_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device):
    # wandb.watch(decoder, criterion, log="all", log_freq=100)
    stdgi.train()
    decoder.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        y_grt = data["Y"].to(device).float()
        x = data["X"].to(device).float()
        G = data["G2"].to(device).float()
        l = data["l"].to(device).float()
        # breakpoint()
        h = stdgi.encoder.inductive(x,l.squeeze(),G)
        y_prd = decoder(h)
        # breakpoint()
        batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    train_loss = epoch_loss / len(dataloader)
    return train_loss
