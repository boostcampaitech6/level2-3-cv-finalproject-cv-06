import json
import os
import os.path as osp
from collections import defaultdict as dd

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class NormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset",
    ):
        super().__init__()
        self.sequence_length = sequence_length

        self.scaler = MinMaxScaler()

        # Load the dataset
        file_list = os.listdir(root)

        df_list = []

        self.length = 0
        self.range_table = []

        self.real_length = 0
        self.real_idx_table = []

        for i, file_name in enumerate(file_list):
            dat = pd.read_csv(root + "/" + file_name)
            dat.drop(columns=["Frame"], inplace=True)

            print(f"==>>{i}번째 dat.shape: {dat.shape}")

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            for id_to_del in id_counter[id_counter < sequence_length].index:
                dat.drop(dat[dat["ID"] == id_to_del].index, inplace=True)

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            print(f"==>>{i}번째 처리 후 dat.shape: {dat.shape}")
            assert len(id_counter[id_counter < sequence_length].index) == 0

            for count in id_counter:
                cur_id_length = count - sequence_length + 1
                self.range_table.append(self.length + cur_id_length)
                self.real_idx_table.append(self.real_length + count)
                self.length += cur_id_length
                self.real_length += count

            dat["ID"] = dat["ID"].astype("str") + f"_{i}"
            df_list.append(dat.copy())

        self.dat = pd.concat(df_list, ignore_index=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.find_real_idx(idx)

        sequence = self.dat[real_idx : real_idx + self.sequence_length].copy()
        sequence.drop(columns=["ID"], inplace=True)
        sequence = self.scaler.fit_transform(sequence.values)
        sequence = np.array(sequence)

        return torch.from_numpy(sequence).float()

    def find_real_idx(self, idx):

        start = 0
        end = len(self.range_table) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.range_table[mid] == idx:
                real_idx = idx + ((mid + 1) * (self.sequence_length - 1))
                return real_idx

            if self.range_table[mid] > idx:
                end = mid - 1
            else:
                start = mid + 1

        real_idx = idx + (start * (self.sequence_length - 1))

        return real_idx


class AbnormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/abnormal",
        label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/label",
    ):
        super().__init__()
        self.sequence_length = sequence_length

        self.scaler = MinMaxScaler()
        # 데이터 값 [0,1] 범위로 scaling할때 사용

        # Load the dataset
        file_list = os.listdir(root)

        df_list = []

        self.length = 0
        self.range_table = []

        self.real_length = 0
        self.real_idx_table = []

        for i, file_name in enumerate(file_list):
            dat = pd.read_csv(root + "/" + file_name)
            # dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

            print(f"==>>{i}번째 dat.shape: {dat.shape}")

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            for id_to_del in id_counter[id_counter < sequence_length].index:
                dat.drop(dat[dat["ID"] == id_to_del].index, inplace=True)

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            print(f"==>>{i}번째 처리 후 dat.shape: {dat.shape}")
            assert len(id_counter[id_counter < sequence_length].index) == 0

            for count in id_counter:
                cur_id_length = count - sequence_length + 1
                self.range_table.append(self.length + cur_id_length)
                self.real_idx_table.append(self.real_length + count)
                self.length += cur_id_length
                self.real_length += count

            dat["ID"] = dat["ID"].astype("str") + f"_{i}"
            df_list.append(dat.copy())

        self.dat = pd.concat(df_list, ignore_index=True)

        # 정답 frame 담은 dict 만들기
        self.frame_label = dd(lambda: dd(lambda: [-1, -1]))

        folder_list = os.listdir(label_root)

        for folder in folder_list:
            json_list = os.listdir(label_root + "/" + folder)

            for js in json_list:
                with open(label_root + "/" + folder + "/" + js, "r") as j:
                    json_dict = json.load(j)

                for dict in json_dict["annotations"]["track"]:
                    if dict["@label"].endswith("_start"):
                        cur_id = dict["@id"]
                        self.frame_label[js[:-5]][cur_id][0] = dict["box"][0]["@frame"]
                    elif dict["@label"].endswith("_end"):
                        cur_id = dict["@id"]
                        self.frame_label[js[:-5]][cur_id][1] = dict["box"][0]["@frame"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.find_real_idx(idx)

        sequence = self.dat[real_idx : real_idx + self.sequence_length].copy()
        target_frames = sequence["Frame"].values
        target_filename = sequence["Filename"].unique()[0].split(".")[0]
        sequence.drop(columns=["ID"], inplace=True)
        sequence.drop(columns=["Frame"], inplace=True)
        sequence.drop(columns=["Filename"], inplace=True)
        # sequence = self.scaler.fit_transform(sequence.values)
        sequence = np.array(sequence)

        target_labels = []

        for target_frame in target_frames:
            temp = 0
            for cur_id in range(0, len(self.frame_label[target_filename].keys()), 2):
                if int(target_frame) >= int(
                    self.frame_label[target_filename][str(int(cur_id))][0]
                ) and int(target_frame) <= int(
                    self.frame_label[target_filename][str(int(cur_id) + 1)][1]
                ):
                    temp = 1

            target_labels.append(temp)

        target_labels = torch.LongTensor(target_labels)

        return (sequence, target_labels)

    def find_real_idx(self, idx):

        start = 0
        end = len(self.range_table) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.range_table[mid] == idx:
                real_idx = idx + ((mid + 1) * (self.sequence_length - 1))
                return real_idx

            if self.range_table[mid] > idx:
                end = mid - 1
            else:
                start = mid + 1

        real_idx = idx + (start * (self.sequence_length - 1))

        return real_idx
