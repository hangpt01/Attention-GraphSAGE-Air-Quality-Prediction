import torch 
from tqdm import tqdm 

def train_stdgi(
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2
):
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
            x = data["X"].to(device).float()
            G = data["G"][:, :, :, :, 0].to(device).float()
            output = stdgi(x, x, G)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            d_loss = criterion(output, lbl)
            d_loss.backward()
            optim_d.step()

        optim_e.zero_grad()
        x = data["X"].to(device).float()
        G = data["G"][:, :, :, :, 0].to(device).float()
        output = stdgi(x, x, G)
        lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
        lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
        lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
        e_loss = criterion(output, lbl)
        e_loss.backward()
        optim_e.step()
        epoch_loss += e_loss.detach().cpu().item()
    return epoch_loss / len(dataloader)
