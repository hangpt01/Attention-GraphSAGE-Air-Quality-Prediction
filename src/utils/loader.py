# from builtins import breakpoint
from builtins import breakpoint
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import (
    Optional,
    Dict,
    Any,
    Union,
    List,
    Iterable,
    Tuple,
    NamedTuple,
    Callable,
)
import math
import torch
import random
import numpy as np
import pandas as pd
import geopy.distance
from os import listdir
from os.path import isfile, join


def get_distance(coords_1, coords_2):
    return geopy.distance.geodesic(coords_1, coords_2).km


def get_distance_matrix(list_col_train_int, location):
    m_ = []
    for target_station in list_col_train_int:
        matrix = []
        for i in list_col_train_int:
            matrix.append(get_distance(location[i], location[target_station]))
        res = np.array(matrix)
        m_.append(res)
    return np.array(m_)


# chi su dung data PM truoc
def get_columns(file_path):
    """
    Get List Stations
    Return Dict {"Numerical Name": "Japanese Station Name"}
    """
    fl = file_path + "PM2.5.csv"
    df = pd.read_csv(fl)
    df = df.fillna(5)
    cols = df.columns.to_list()
    res, res_rev = {}, {}
    for i, col in enumerate(cols):
        if i == 0:
            pass
        else:
            i -= 1
            stat_name = "Station_" + str(i)
            res.update({stat_name: col})
            res_rev.update({col: stat_name})

    pm_df = df.rename(columns=res_rev)
    return res, res_rev, pm_df


def preprocess_pipeline(df, args, scaler=None):
    (a, b, c) = df.shape
    res = np.reshape(df, (-1, c))
    # threshold_ = [90, 90, 30]
    for i in range(c):
        threshold = np.percentile(res[:, i], 95)
        # threshold = threshold_[i]
        res[:, i] = np.where(res[:, i] > threshold, threshold, res[:, i])
    if scaler == None:
        scaler = MinMaxScaler((-1, 1))
        res_ = scaler.fit_transform(res)
    else:
        res_ = scaler.transform(res)
    res_aq = res_.copy()
    res_climate = res_.copy()
    if args.use_wind:
        res_aq[:, -1] = res[:, -1]
    res_aq = np.reshape(res_aq, (-1, b, c))
    res_climate = np.reshape(res_climate, (-1, b, c))
    trans_df = res_aq[:, :, :]
    idx_climate = len(args.features)
    climate_df = res_climate[
        :, :, idx_climate:
    ]
    del res_aq
    del res_climate
    del res
    return trans_df, climate_df, scaler


def get_list_file(folder_path):
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return onlyfiles


def comb_df(file_path, pm_df, res):
    list_file = get_list_file(file_path)
    list_file.remove("PM2.5.csv")
    list_file.remove("location.csv")
    column = [res[i] for i in list(pm_df.columns)[1:]]
    comb_arr = pm_df.iloc[:, 1:].to_numpy()
    comb_arr = np.expand_dims(comb_arr, -1)
    for file_name in list_file:
        df = pd.read_csv(file_path + file_name)
        # preprocess()
        df = df.ffill()
        df = df.fillna(5)
        df = df[column]
        arr = df.to_numpy()
        arr = np.expand_dims(arr, -1)
        comb_arr = np.concatenate((comb_arr, arr), -1)
    del arr
    return comb_arr, column


from torch.utils.data import Dataset


def location_arr(file_path, res):
    location_df = pd.read_csv(file_path + "location.csv")
    list_location = []
    for i in res.keys():
        loc = location_df[location_df["location"] == res[i]].to_numpy()[0, 1:]
        list_location.append([loc[1], loc[0]])
    del loc
    return np.array(list_location)


def get_data_array(args, file_path):
    columns1 = args.features
    location_df = pd.read_csv(f"{file_path}location.csv")
    station = location_df["station"].values
    location = location_df[["longitude", "latitude"]].values
    location_ = location[:, [1, 0]]
    distance_matrix = get_distance_matrix(range(len(station)), location_)
    distance_matrix += np.identity(len(station)) * 7
    list_arr = []
    list_corr = []
    for i in station:
        df = pd.read_csv(file_path + f"{i}.csv")[columns1]
        df = df.fillna(method="ffill")
        df = df.fillna(10)
        arr = df.astype(float).values
        arr = np.expand_dims(arr, axis=1)
        list_arr.append(arr)
    list_arr = np.concatenate(list_arr, axis=1)
    pm2_5 = list_arr[:, :, 0]
    # breakpoint()
    corr = pd.DataFrame(pm2_5).corr().values
    for i in range(list_arr.shape[-1]):
        pm2_5 = list_arr[:, :, i]
        corr = pd.DataFrame(pm2_5).corr().values
        list_corr.append(corr)
    del df
    del arr
    del location
    del location_df
    return list_arr, location_, station, columns1, distance_matrix, list_corr


def convert_2_point_coord_to_direction(coords1, coords2):
    x_dest, y_dest = coords1
    x_target, y_target = coords2

    deltaX = x_target - x_dest
    deltaY = y_target - y_dest

    degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180

    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp
    compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    compass_lookup = round(degrees_final / 45)
    return compass_brackets[compass_lookup], degrees_final


class AQDataSet(Dataset):
    def __init__(
        self,
        data_df,
        climate_df,
        location_df,
        dist_matrix,
        test_station=None,
        test=False,
        valid=False,
        corr_matrix=None,
        args=None,
    ) -> None:
        super().__init__()
        assert not (test and test_station == None), "pha test yeu cau nhap tram test"
        assert not (
            test_station in args.train_station
        ), "tram test khong trong tram train"
        self.list_cols_train_int = args.train_station
        self.sequence_length = args.sequence_length
        self.test = test
        self.valid = valid
        self.dist_matrix = dist_matrix
        self.data_df = data_df
        self.location = location_df
        self.climate_df = climate_df
        self.window_size = 0
        self.corr_matrix = corr_matrix
        self.train_cpt = args.train_pct
        self.valid_cpt = args.valid_pct
        self.test_cpt = args.test_pct
        self.dist_threshold = args.dist_threshold
        self.corr_threshold = args.corr_threshold
        idx_test = int(len(data_df) * (1 - self.test_cpt))

        # phan data train thi khong lien quan gi den data test
        self.X_train = data_df[:idx_test, :, :]
        self.climate_train = climate_df[:idx_test, :, :]
        self.type_G = args.type_g
        self.adj_ = np.where(dist_matrix < self.dist_threshold,1,0)

        # test data
        if self.test:
            # phan data test khong lien quan gi data train
            test_station = int(test_station)
            self.test_station = test_station
            lst_cols_input_test_int = list(
                set(self.list_cols_train_int) - set([self.list_cols_train_int[-1]])
            )
            self.X_test = data_df[idx_test:, lst_cols_input_test_int, :]
            self.l_test = self.get_inverse_distance_matrix(
                lst_cols_input_test_int, test_station
            )
            self.Y_test = data_df[idx_test:, test_station, :]
            self.climate_test = climate_df[idx_test:, test_station, :]
            self.G_test,self.nb_adj = self.get_adjacency_matrix(lst_cols_input_test_int)
        elif self.valid:
            # phan data test khong lien quan gi data train
            test_station = int(test_station)
            self.test_station = test_station
            lst_cols_input_test_int = list(
                set(self.list_cols_train_int) - set([self.list_cols_train_int[-1]])
            )
            self.X_test = data_df[:idx_test, lst_cols_input_test_int, :]
            # convert data gio theo target station
            self.l_test = self.get_inverse_distance_matrix(
                lst_cols_input_test_int, test_station
            )
            self.Y_test = data_df[:idx_test, test_station, :]
            self.climate_test = climate_df[:idx_test, test_station, :]
            self.G_test,self.nb_adj = self.get_adjacency_matrix(lst_cols_input_test_int)

    def get_list_angles(self, test_stat, list_stat):
        target_stat = tuple(self.location[test_stat, :])
        angles = []
        for stat in list_stat:
            source_stat = tuple(self.location[stat, :])
            angle = convert_2_point_coord_to_direction(source_stat, target_stat)
            angles.append(angle[1])
        return angles

    def get_adjacency_matrix(self, list_station):
        """
        G1: Sử dụng IDW
        G2: Sử dụng IDW và corr_threshold
        G3: Sử dụng  IDW và dist_threshold
        G4: Sử dụng IDW và 2 threshold
        """

        distance_adj = self.dist_matrix[np.ix_(list_station, list_station)]
        nn_adj = self.adj_[np.ix_(list_station, list_station)]
        inverse_distance_adj = 1 / distance_adj
        if self.corr_matrix is not None:
            corr_adj = self.corr_matrix[np.ix_(list_station, list_station)]
        if self.type_G == 1:
            inverse_distance_adj_ = inverse_distance_adj
        elif self.type_G == 2:
            inverse_distance_adj_ = np.where(
                corr_adj > self.corr_threshold, inverse_distance_adj, 0
            )
        elif self.type_G == 3:
            inverse_distance_adj_ = np.where(
                inverse_distance_adj > 1 / self.dist_threshold, inverse_distance_adj, 0
            )
        elif self.type_G == 4:
            inverse_distance_adj_ = np.where(
                corr_adj > self.corr_threshold, inverse_distance_adj, 0
            )
            inverse_distance_adj_ = np.where(
                inverse_distance_adj_ > 1 / self.dist_threshold, inverse_distance_adj_, 0
            )
        adj = inverse_distance_adj_ / inverse_distance_adj_.sum(axis=-1, keepdims=True)
        return adj,nn_adj

    def get_inverse_distance_matrix(self, list_station, target_station):
        dis_vec = self.dist_matrix[np.ix_([target_station], list_station)]
        inverse_dis_vec = 1 / dis_vec
        inverse_dis_vec = inverse_dis_vec / inverse_dis_vec.sum(axis=-1)
        return inverse_dis_vec


    def __getitem__(self, index: int):
        if self.test or self.valid:
            idx = index
        else:
            n_samples = (
                self.X_train.shape[0] - (self.sequence_length) - self.window_size
            )
            idx = index % n_samples
            picked_target_station_int = self.list_cols_train_int[index // n_samples]
        list_G = []
        if self.test:
            x = self.X_test[idx : idx + self.sequence_length, :]
            x_k = self.X_test[
                idx + self.window_size : idx + self.sequence_length + self.window_size,
                :,
            ]
            y = self.Y_test[idx + self.sequence_length - 1 + self.window_size, 0]
            G = self.G_test
            nb_adj = self.nb_adj
            l = self.l_test
            climate = self.climate_test[
                idx + self.sequence_length - 1 + self.window_size, :
            ]
            list_G = [G]
        elif self.valid:
            x = self.X_test[idx : idx + self.sequence_length, :]
            x_k = self.X_test[
                idx + self.window_size : idx + self.sequence_length + self.window_size,
                :,
            ]
            nb_adj = self.nb_adj
            y = self.Y_test[idx + self.sequence_length - 1 + self.window_size, 0]
            G = self.G_test
            l = self.l_test
            climate = self.climate_test[
                idx + self.sequence_length - 1 + self.window_size, :
            ]
            list_G = [G]
        else:
            # chon 1 tram ngau  nhien trong 28 tram lam target tai moi sample
            lst_col_train_int = list(
                set(self.list_cols_train_int) - set([picked_target_station_int])
            )
            x = self.X_train[idx : idx + self.sequence_length, lst_col_train_int, :]
            x_k = self.X_train[
                idx + self.window_size : idx + self.sequence_length + self.window_size,
                lst_col_train_int,
                :,
            ]

            y = self.X_train[
                idx + self.sequence_length - 1 + self.window_size,
                picked_target_station_int,
                0,
            ]
            climate = self.climate_train[
                idx + self.sequence_length - 1 + self.window_size,
                picked_target_station_int,
                :,
            ]
            G,nb_adj = self.get_adjacency_matrix(lst_col_train_int)
            list_G = [G]
            l = self.get_inverse_distance_matrix(
                lst_col_train_int, picked_target_station_int
            )

        sample = {
            "X": x,
            "Y": np.array([y]),
            "l": np.array(l),
            "climate": climate,
            "X_k": x_k,
            "G2": nb_adj,
        }
        sample["G"] = np.stack(list_G, -1)
        return sample

    def __len__(self) -> int:
        if self.test:
            return self.X_test.shape[0] - self.sequence_length - self.window_size
        elif self.valid:
            return self.X_train.shape[0] - (self.sequence_length) - self.window_size
        else:
            return (
                self.X_train.shape[0] - (self.sequence_length) - self.window_size
            ) * len(self.list_cols_train_int)
