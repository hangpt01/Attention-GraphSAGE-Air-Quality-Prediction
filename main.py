import argparse
from builtins import breakpoint
from distutils.util import copydir_run_2to3
from src.models.stdgi import STDGI
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from src.utils.utils import config_seed, load_model, EarlyStopping
from src.utils.loader import get_data_array, preprocess_pipeline, AQDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=52, type=int, help="Seed")
    parser.add_argument(
        "--train_station",
        default=[i for i in range(8)],
        type=list,
    )
    parser.add_argument(
        "--test_station",
        default=[i for i in range(8, 12, 1)],
        type=list,
    )
    # Config STDGI
    parser.add_argument("--input_dim", default=9, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--sequence_length", default=12, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--lr_stdgi", default=0.001, type=float)
    parser.add_argument("--num_epochs_stdgi", default=100, type=int)
    parser.add_argument("--output_stdgi", default=60, type=int)
    parser.add_argument("--en_hid1", default=64, type=int)
    parser.add_argument("--en_hid2", default=128, type=int)
    parser.add_argument("--dis_hid", default=6, type=int)
    parser.add_argument("--act_fn", default="relu", type=str)
    parser.add_argument("--delta_stdgi", default=0, type=float)
    parser.add_argument("--stdgi_noise_min", default=0.4, type=float)
    parser.add_argument("--stdgi_noise_max", default=0.7, type=float)
    # Config Decoder
    parser.add_argument("--train_pct", default=0.6, type=float)
    parser.add_argument("--valid_pct", default=0.25, type=float)
    parser.add_argument("--test_pct", default=0.15, type=float)
    parser.add_argument("--use_wind", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--log_wandb", action="store_false")
    parser.add_argument(
        "--climate_features",
        default=[
            "2m_temperature",
            "surface_pressure",
            "evaporation",
            "total_precipitation",
        ],
        type=list,
    )
    parser.add_argument("--features",default='PM2.5,wind_speed',type=str)
    parser.add_argument(
        "--model_type", type=str, choices=["gede", "wogcn", "wornnencoder"],default='gede'
    )
    # parser.add_argument("--group_name", type=str, default="", required=True)
    parser.add_argument("--dataset", type=str, choices=["beijing", "uk","hanoi"],default="beijing")
    return parser.parse_args()

from src.modules.train import train_stdgi
from src.utils.loader import AQDataSet
import logging
import wandb
import json
import os 
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    args = parse_args()

    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg))

    config_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.group_name = "Tuning_Features_merged_AQ_{}".format(args.dataset)
    args.features = args.features.split(",")
    args.idx_climate = len(args.features)
    if args.dataset == "uk":
        file_path = "./data/uk_AQ/"
    elif args.dataset == "beijing":
        file_path = "./data/beijing_AQ/"
    elif args.dataset == "hanoi":
        file_path = "./data/AQ_hanoi/"
        args.climate_features = ["humidity", "temperature"]
        args.use_wind = False  # Hanoi has no wind data
    if args.use_wind:
        args.climate_features = [
            "2m_temperature",
            "surface_pressure",
            "evaporation",
            "total_precipitation",
            "wind_speed",
            "wind_angle",
        ]

    comb_arr, location_, station, features_name, corr = get_data_array(args, file_path)
    args.input_dim = len(features_name)
    # print(station)
    trans_df, climate_df, scaler = preprocess_pipeline(comb_arr, args)
    # breakpoint()
    config["features"] = features_name
    test_name = "test1"
    if args.dataset == "beijing":
        args.train_station = [18,11,3,15,8,1,9]
        args.valid_station = [12,7,2,10,13]
        args.test_station = [0, 4, 5, 6]
    elif args.dataset == "uk":
        args.train_station = [15,17,19,21,48,73,96,114,131]
        args.valid_station = [20,34,56,85]
        args.test_station = [97, 98, 134, 137]
    elif args.dataset == "hanoi":
        args.train_station = [55, 53, 69, 49, 10, 37, 64, 41, 45, 19, 60, 2, 7, 40, 32, 52, 54, 17, 0, 18, 35, 56, 67, 33, 22, 44, 61, 30, 72, 16, 65, 24, 39, 29, 71,6, 74, 58,36, 5, 9, 70]
        args.valid_station = [50, 46, 62, 31, 14, 25, 11, 26, 3, 66, 68, 63, 57, 20, 8, 34, 21,42, 13,43, 73]
        args.test_station = [1, 4, 38, 12, 47, 48, 51, 23, 28,59,27, 15]
    corr = None
    args.num_input_station = len(args.train_station) - 1
    train_dataset = AQDataSet(
        data_df=trans_df[:],
        climate_df=climate_df,
        location_df=location_,
        list_train_station=args.train_station,
        input_dim=args.sequence_length,
        corr=corr,
        args=args,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # config["loss"] = 'mse'
    args.name = (
        f"{args.model_type}__{args.dataset}_{args.seed}_{args.name}"
    )
    if args.log_wandb:
        wandb.init(
            entity="aiotlab",
            project="Spatial_PM2.5",
            group="Features_merged_AQ_{}".format(args.dataset),
            name=f"{args.name}",
            config=config,
        )

    # Model Stdgi
    stdgi = STDGI(
        in_ft=args.input_dim,
        out_ft=args.output_stdgi,
        en_hid1=args.en_hid1,
        en_hid2=args.en_hid2,
        dis_hid=args.dis_hid,
        stdgi_noise_min=args.stdgi_noise_min,
        stdgi_noise_max=args.stdgi_noise_max,
        model_type=args.model_type,
        num_input_station=args.num_input_station,
    ).to(device)
    l2_coef = 0.0
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    stdgi_optimizer_e = torch.optim.Adam(
        stdgi.encoder.parameters(), lr=args.lr_stdgi, weight_decay=l2_coef
    )
    stdgi_optimizer_d = torch.optim.Adam(
        stdgi.disc.parameters(), lr=args.lr_stdgi, weight_decay=l2_coef
    )
    if not os.path.exists(f"output/{args.group_name}/checkpoint/"):
        print(f"Make dir output/{args.group_name}/checkpoint/ ...")
        os.makedirs(f"output/{args.group_name}/checkpoint/")
    early_stopping_stdgi = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_stdgi,
        path=f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt",
    )
    # scheduler_stdgi = ReduceLROnPlateau(stdgi_optimizer, "min", factor=0.5, patience=3)

    logging.info(
        f"Training stdgi || attention decoder || epochs {args.num_epochs_stdgi} || lr {args.lr_stdgi}"
    )
    train_stdgi_loss = []
    for i in range(args.num_epochs_stdgi):
        if not early_stopping_stdgi.early_stop:
            loss = train_stdgi(
                stdgi,
                train_dataloader,
                stdgi_optimizer_e,
                stdgi_optimizer_d,
                bce_loss,
                device,
                n_steps=2,
            )
            early_stopping_stdgi(loss, stdgi)
            # scheduler_stdgi.step(loss)
            if args.log_wandb:
                wandb.log({"loss/stdgi_loss": loss})
            logging.info("Epochs/Loss: {}/ {}".format(i, loss))
    if args.log_wandb:
        wandb.run.summary["best_loss_stdgi"] = early_stopping_stdgi.best_score
    load_model(stdgi, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")

