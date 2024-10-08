import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)
def mdape(y_true, y_pred):
	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100

def cal_acc(y_prd, y_grt):
    mae = mean_absolute_error(y_grt, y_prd)
    mse = mean_squared_error(y_grt, y_prd, squared=True)
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    rmse = mean_squared_error(y_grt, y_prd, squared=False)
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    mdape_ = mdape(y_grt,y_prd)
    return mae, mse, mape, rmse, r2, corr,mdape_


def test_atten_decoder_fn(
    stdgi, decoder, dataloader, device,criterion, scaler=None,test=True, args=None
):
    decoder.eval()
    stdgi.eval()
    
    list_prd = []
    list_grt = []
    # breakpoint()
    epoch_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            batch_loss = 0
            y_grt = data["Y"].to(device).float()
            x = data["X"].to(device).float()
            G = data["G2"].to(device).float()
            l = data["l"].to(device).float()
            h = stdgi.encoder.inductive(x,l.squeeze(), G)
            y_prd = decoder(h) 
            batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
            y_prd = torch.squeeze(y_prd).cpu().detach().numpy().tolist()
            y_grt = torch.squeeze(y_grt).cpu().detach().numpy().tolist()
            list_prd += y_prd
            list_grt += y_grt
            epoch_loss += batch_loss.item()
    if test:
        # breakpoint()
        a_max = scaler.data_max_[0]
        a_min = scaler.data_min_[0]
        min_ = scaler.feature_range[0]
        max_ = scaler.feature_range[1]
        scale_ = scaler.scale_[0]
        list_grt = (np.array(list_grt) - min_) /scale_ + a_min
        list_prd = (np.array(list_prd) - min_) /scale_ + a_min
        # list_grt = (np.array(list_grt) + 1) / 2 * (a_max - a_min) + a_min
        # list_prd = (np.array(list_prd) + 1) / 2 * (a_max - a_min) + a_min
        list_grt_ = [float(i) for i in list_grt]
        list_prd_ = [float(i) for i in list_prd]
        return list_prd_, list_grt_, epoch_loss / len(dataloader)
    else:
        return epoch_loss / len(dataloader)
