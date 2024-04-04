import json
import os
import os.path as osp
from collections import defaultdict as dd

import numpy as np
import pandas as pd
import torch

# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


class NormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        prediction_time=1,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv/normal/val",
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_time = prediction_time

        # self.scaler = MinMaxScaler()
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
            dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

            print(f"==>>{i}번째 dat.shape: {dat.shape}")

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            for id_to_del in id_counter[
                id_counter < sequence_length + prediction_time
            ].index:
                dat.drop(dat[dat["ID"] == id_to_del].index, inplace=True)

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            print(f"==>>{i}번째 처리 후 dat.shape: {dat.shape}")
            assert (
                len(id_counter[id_counter < sequence_length + prediction_time].index)
                == 0
            )

            for count in id_counter:
                cur_id_length = count - sequence_length - prediction_time + 1
                self.range_table.append(self.length + cur_id_length)
                self.real_idx_table.append(self.real_length + count)
                self.length += cur_id_length
                self.real_length += count

            dat["ID"] = dat["ID"].astype("str") + f"_{i}"
            df_list.append(dat.copy())

        self.dat = pd.concat(df_list, ignore_index=True)
        # self.dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

        id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)

        # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # DF를 다 합치고 나서 ID를 거르면 데이터셋 초기화에 1분정도 걸리지만
        # DF 조각마다 ID를 거르고나서 합치면 6초 밖에 안 걸린다
        # self.checker = []

        # for id_to_del in id_counter[id_counter < sequence_length + prediction_time].index:
        #     self.checker.append((id_to_del, id_counter[id_to_del]))

        #     self.dat.drop(self.dat[self.dat["ID"] == id_to_del].index, inplace=True)

        # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # # sequence_length + prediction_time 보다 짧은 ID를 지우는 것을 한번만 하면
        # # 13개 ID가 sequence_length + prediction_time보다 짧은데도 남아 있다???
        # id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)

        # if len(id_counter[id_counter < sequence_length + prediction_time].index) != 0:
        #     for id_to_del in id_counter[id_counter < sequence_length + prediction_time].index:
        #         self.checker.append((id_to_del, id_counter[id_to_del]))

        #         self.dat.drop(self.dat[self.dat["ID"] == id_to_del].index, inplace=True)

        # id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)
        # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        assert (
            len(id_counter[id_counter < sequence_length + prediction_time].index) == 0
        )

        # self.length = 0

        # self.range_table = []

        # for count in id_counter:
        #     cur_id_length = count - sequence_length - prediction_time + 1
        #     self.range_table.append(self.length + cur_id_length)
        #     self.length += cur_id_length

        # self.dat.drop(columns=["ID"], inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.find_real_idx(idx)

        sequence = self.dat[real_idx : real_idx + self.sequence_length].copy()
        sequence.drop(columns=["ID"], inplace=True)
        sequence = np.array(sequence)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # sequence = self.scaler.fit_transform(sequence.values)
        # # 데이터 값 [min, max] -> [0,1] 범위로 scaling
        # scale 된 후에는 numpy array로 변환된다
        # sequence나 target은 이미 yolo v8에서 xywhn, xyn으로 0~1 범위인데 scaling을 한번 더 할 필요가 있을지?
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # (self.sequence_length, 38)
        target = self.dat[
            real_idx
            + self.sequence_length : real_idx
            + self.sequence_length
            + self.prediction_time
        ].copy()
        target.drop(columns=["ID"], inplace=True)
        target = np.array(target)
        # target = self.scaler.fit_transform(target.values)
        # (self.prediction_time, 38)

        label = torch.LongTensor([0 for i in range(self.prediction_time)])

        return (
            torch.from_numpy(sequence).float(),
            torch.from_numpy(target).float(),
            label,
        )

    def find_real_idx(self, idx):

        start = 0
        end = len(self.range_table) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.range_table[mid] == idx:
                real_idx = idx + (
                    (mid + 1) * (self.sequence_length + self.prediction_time - 1)
                )
                return real_idx

            if self.range_table[mid] > idx:
                end = mid - 1
            else:
                start = mid + 1

        real_idx = idx + (start * (self.sequence_length + self.prediction_time - 1))

        return real_idx


class AbnormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        prediction_time=1,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv/abnormal/val",
        label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal/val",
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_time = prediction_time

        # self.scaler = MinMaxScaler()
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

            for id_to_del in id_counter[
                id_counter < sequence_length + prediction_time
            ].index:
                dat.drop(dat[dat["ID"] == id_to_del].index, inplace=True)

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            print(f"==>>{i}번째 처리 후 dat.shape: {dat.shape}")
            assert (
                len(id_counter[id_counter < sequence_length + prediction_time].index)
                == 0
            )

            for count in id_counter:
                cur_id_length = count - sequence_length - prediction_time + 1
                self.range_table.append(self.length + cur_id_length)
                self.real_idx_table.append(self.real_length + count)
                self.length += cur_id_length
                self.real_length += count

            dat["ID"] = dat["ID"].astype("str") + f"_{i}"
            df_list.append(dat.copy())

        self.dat = pd.concat(df_list, ignore_index=True)
        # self.dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

        id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)

        assert (
            len(id_counter[id_counter < sequence_length + prediction_time].index) == 0
        )

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # TODO: 한 영상에 start end 여러번 있는 경우 고려해서 코드 수정하기
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
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.find_real_idx(idx)

        sequence = self.dat[real_idx : real_idx + self.sequence_length].copy()
        sequence.drop(columns=["ID"], inplace=True)
        sequence.drop(columns=["Frame"], inplace=True)
        sequence.drop(columns=["Filename"], inplace=True)
        sequence = np.array(sequence)
        # (self.sequence_length, 38)
        target = self.dat[
            real_idx
            + self.sequence_length : real_idx
            + self.sequence_length
            + self.prediction_time
        ].copy()
        target_frames = target["Frame"].unique()
        target_filename = target["Filename"].unique()[0].split(".")[0]

        target.drop(columns=["ID"], inplace=True)
        target.drop(columns=["Frame"], inplace=True)
        target.drop(columns=["Filename"], inplace=True)
        target = np.array(target)
        # target = self.scaler.fit_transform(target.values)
        # (self.prediction_time, 38)

        target_labels = []

        for target_frame in target_frames:
            temp = 0
            for cur_id in self.frame_label[target_filename].keys():
                if int(target_frame) >= int(
                    self.frame_label[target_filename][cur_id][0]
                ) and int(target_frame) <= int(
                    self.frame_label[target_filename][cur_id][1]
                ):
                    temp = 1

            target_labels.append(temp)

        target_labels = torch.LongTensor(target_labels)

        return (
            torch.from_numpy(sequence).float(),
            torch.from_numpy(target).float(),
            target_labels,
        )

    def find_real_idx(self, idx):

        start = 0
        end = len(self.range_table) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.range_table[mid] == idx:
                real_idx = idx + (
                    (mid + 1) * (self.sequence_length + self.prediction_time - 1)
                )
                return real_idx

            if self.range_table[mid] > idx:
                end = mid - 1
            else:
                start = mid + 1

        real_idx = idx + (start * (self.sequence_length + self.prediction_time - 1))

        return real_idx


class NormalVMAE(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        # is_train=1,
        model_size="small",
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/normal",
        # label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal",
    ):
        super().__init__()
        # self.is_train = is_train
        # normal의 경우 torch.utils.data.random_split 함수로 train/val 나눔

        self.path = root

        folder_list = os.listdir(self.path)
        folder_list.sort()

        self.data_list = []

        for folder_name in folder_list:
            if folder_name.endswith("_base") and model_size == "small":
                continue
            elif not folder_name.endswith("_base") and model_size != "small":
                continue
            print(f"==>> {folder_name} 폴더 데이터 로딩 시작")

            folder_path = folder_name + "/"
            data_list = os.listdir(self.path + "/" + folder_path)
            data_list.sort()
            data_list = [folder_path + name for name in data_list]
            self.data_list.extend(data_list)
            print(f"==>> {folder_name} 폴더 데이터 로딩 완료")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        feature = np.load(self.path + "/" + file_name)
        # 정상 영상 feature는 (57,710) 또는 (38,710)
        if feature.shape[0] % 12 == 0:
            feature_npy = feature
        else:
            count = (feature.shape[0] // 12) + 1

            feature_npy = np.zeros((count * 12, 710))
            # 12로 나눌 수 있도록 (60, 710) or (48, 710) 준비

            feature_npy[: feature.shape[0]] = feature
            # np.load로 불러온 정상영상 feature는 (57, 710) 또는 (38,710)

            feature_npy[feature.shape[0] :] = [feature_npy[-1]] * (
                count * 12 - feature.shape[0]
            )
            # 정상영상 feature의 마지막 부분으로 빈 자리 채우기

        feature_npy = feature_npy.reshape(12, -1, 710)
        # (12, 5, 710) or (12, 4, 710)
        feature_npy = np.mean(feature_npy, axis=1)
        # 이상행동 영상 feature의 (12,710)과 같아지도록 평균으로 조절

        gts = np.zeros(11)
        # 정상영상은 전부 정답이 0

        return (
            torch.from_numpy(feature_npy[:-1, :]).float(),
            torch.from_numpy(gts).float(),
        )


class AbnormalVMAE(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        is_train=1,
        model_size="small",
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/abnormal",
        label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal",
    ):
        print(f"==>> abnormal 데이터 로딩 시작")
        super().__init__()
        self.is_train = is_train

        if self.is_train == 1:
            self.path = root + "/train/"
            if model_size == "small":
                self.label_path = label_root + "/train/abnormal_train.json"
            else:
                self.label_path = label_root + "/train/abnormal_train_base.json"
        else:
            self.path = root + "/val/"
            if model_size == "small":
                self.label_path = label_root + "/val/abnormal_val.json"
            else:
                self.label_path = label_root + "/val/abnormal_val_base.json"

        with open(self.label_path, "r", encoding="utf-8") as j:
            self.data_list = json.load(j)
        print(f"==>> abnormal 데이터 로딩 완료")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_info = self.data_list[str(idx)]

        feature_npy = np.load(self.path + file_info["filename"])
        # feature_npy.shape: (12, 710)

        # file_name = file_info["filename"].split("/")[-1].split(".")[0]

        gts = np.zeros(176)
        # 이상행동 영상 180 프레임 => 12 * 16 = 192 가 되도록 길이 연장

        for start, end in zip(file_info["frames_start"], file_info["frames_end"]):
            gts[int(start) - 1 : min(int(end), 176)] = 1

        # for i in range(12):
        #     gts[180 + i] = gts[179]
        # @@ feature extraction할때 마지막 조각에서 frame 개수가 16개가 안되면 마지막 frame을 복사해서 추가함

        if self.is_train:
            gts = gts.reshape(11, 16)
            # (192) => (12, 16)로 변경
            # gts = np.mean(gts, axis=1)
            # 평균 내서 (12)로 변경
            gts = np.max(gts, axis=1)

        # @@ validation일때는 평균내지 않고 (192) numpy array 그대로 반환

        return (
            torch.from_numpy(feature_npy[:-1, :]).float(),
            torch.from_numpy(gts).float(),
        )


class NewNormalVMAE(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        is_train=1,
        model_size="small",
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/UCFCrime/normal",
        # label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal",
        num_segments=200,
        l2_norm=False,
    ):
        set_type = "학습" if is_train == 1 else "검증"
        print(f"==>> normal {set_type} 데이터 로딩 시작")
        super().__init__()
        self.is_train = is_train
        self.l2_norm = l2_norm

        if self.is_train == 1:
            self.path = root + "/train/"
        else:
            self.path = root + "/val/"
        self.num_segments = num_segments

        folder_list = os.listdir(self.path)
        folder_list.sort()

        self.data_list = []

        for folder_name in folder_list:
            if folder_name.endswith("_base") and model_size == "small":
                continue
            elif not folder_name.endswith("_base") and model_size != "small":
                continue
            print(f"==>> {folder_name} 폴더 데이터 로딩 시작")

            folder_path = folder_name + "/"
            data_list = os.listdir(self.path + "/" + folder_path)
            data_list.sort()
            data_list = [folder_path + name for name in data_list]
            self.data_list.extend(data_list)
            print(f"==>> {folder_name} 폴더 데이터 로딩 완료")

        print(f"==>> normal {set_type} 데이터 로딩 완료")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        feature = np.load(self.path + "/" + file_name).astype(np.float32)
        # (원본영상 frame 수 // 16,710)

        if self.l2_norm:
            feature = normalize(feature, norm="l2")

        feature_npy = np.zeros((self.num_segments, 710)).astype(np.float32)

        sample_index = np.linspace(
            0, feature.shape[0], self.num_segments + 1, dtype=np.uint16
        )
        # ex: feature.shape[0]이 62이고, self.num_segments이 200이면
        # sample_index == [ 0  0  0  0  1  1  1  2  2  2  3  3  3  4  4  4  4  5  5  5  6  6  6  7
        #   7  7  8  8  8  8  9  9  9 10 10 10 11 11 11 12 12 12 13 13 13 13 14 14
        #  14 15 15 15 16 16 16 17 17 17 17 18 18 18 19 19 19 20 20 20 21 21 21 22
        #  22 22 22 23 23 23 24 24 24 25 25 25 26 26 26 26 27 27 27 28 28 28 29 29
        #  29 30 30 30 31 31 31 31 32 32 32 33 33 33 34 34 34 35 35 35 35 36 36 36
        #  37 37 37 38 38 38 39 39 39 39 40 40 40 41 41 41 42 42 42 43 43 43 44 44
        #  44 44 45 45 45 46 46 46 47 47 47 48 48 48 48 49 49 49 50 50 50 51 51 51
        #  52 52 52 53 53 53 53 54 54 54 55 55 55 56 56 56 57 57 57 57 58 58 58 59
        #  59 59 60 60 60 61 61 61 62] 꼴
        # ex2: feature.shape[0]이 62이고, self.num_segments이 11이면
        # sample_index == [ 0,  6, 12, 18, 24, 31, 37, 43, 49, 55, 62]

        for i in range(len(sample_index) - 1):
            if sample_index[i] == sample_index[i + 1]:
                feature_npy[i, :] = feature[sample_index[i], :]
            else:
                feature_npy[i, :] = feature[
                    sample_index[i] : sample_index[i + 1], :
                ].mean(0)
                # ex2의 0과 6 => [0:6] => 0~5 feature 6개 평균
                # ex1의 0과 1 => [0:1] => 0~0 feature 1개 평균 => 0번 feature 그대로

        # feature.shape[0]이 self.num_segments보다 짧으면 같은 feature 반복
        # feature.shape[0]이 self.num_segments보다 길면 평균 내서 self.num_segments개로 줄인다

        if self.is_train != 1:
            gts = np.zeros(self.num_segments).astype(np.float32)
            # 정상영상은 전부 정답이 0

            return torch.from_numpy(feature_npy), torch.from_numpy(gts)
        else:
            return torch.from_numpy(feature_npy)


class NewAbnormalVMAE(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        is_train=1,
        model_size="small",
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/UCFCrime/abnormal",
        label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/UCFCrime/test_anomalyv2.txt",
        num_segments=200,
        gt_thr=0.25,
        l2_norm=False,
    ):
        set_type = "학습" if is_train == 1 else "검증"
        print(f"==>> abnormal {set_type} 데이터 로딩 시작")
        super().__init__()
        self.is_train = is_train
        self.l2_norm = l2_norm

        if self.is_train == 1:
            self.path = root + "/train/"
        else:
            self.path = root + "/val/"
            self.label_dict = {}
            with open(label_root, "r", encoding="utf-8") as f:
                for line in f:
                    # line.split()은 ['Arrest/Arrest039_x264.mp4', '15836', '[7215, 10335]\n'] 이런 형태
                    temp = line.split("|")
                    self.label_dict[temp[0].split("/")[1] + ".npy"] = {
                        "frame_counts": int(temp[1]),
                        "frames_gt": temp[2][1:-2].split(","),
                    }

        self.num_segments = num_segments
        self.gt_thr = gt_thr

        folder_list = os.listdir(self.path)
        folder_list.sort()

        self.data_list = []

        for folder_name in folder_list:
            if folder_name.endswith("_base") and model_size == "small":
                continue
            elif not folder_name.endswith("_base") and model_size != "small":
                continue
            print(f"==>> {folder_name} 폴더 데이터 로딩 시작")

            folder_path = folder_name + "/"
            data_list = os.listdir(self.path + "/" + folder_path)
            data_list.sort()
            data_list = [folder_path + name for name in data_list]
            self.data_list.extend(data_list)
            print(f"==>> {folder_name} 폴더 데이터 로딩 완료")

        print(f"==>> abnormal {set_type} 데이터 로딩 완료")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        feature = np.load(self.path + "/" + file_name).astype(np.float32)
        # (원본영상 frame 수 // 16,710)

        if self.l2_norm:
            feature = normalize(feature, norm="l2")

        feature_npy = np.zeros((self.num_segments, 710)).astype(np.float32)

        sample_index = np.linspace(
            0, feature.shape[0], self.num_segments + 1, dtype=np.uint16
        )
        # ex: feature.shape[0]이 62이고, self.num_segments이 11이면
        # sample_index == [ 0,  6, 12, 18, 24, 31, 37, 43, 49, 55, 62]

        for i in range(len(sample_index) - 1):
            if sample_index[i] == sample_index[i + 1]:
                feature_npy[i, :] = feature[sample_index[i], :]
            else:
                feature_npy[i, :] = feature[
                    sample_index[i] : sample_index[i + 1], :
                ].mean(0)
                # ex의 0과 6 => [0:6] => 0~5 feature 6개 평균

        # feature.shape[0]이 self.num_segments보다 짧으면 같은 feature 반복
        # feature.shape[0]이 self.num_segments보다 길면 평균 내서 self.num_segments개로 줄인다

        if self.is_train != 1:
            label_info = self.label_dict[file_name.split("/")[1]]
            frame_counts = label_info["frame_counts"]
            frames_gt = label_info["frames_gt"]

            # if frame_counts % 16 == 0:
            #     gts = np.zeros(frame_counts)
            # else:
            #     gts = np.zeros(frame_counts + (16 - (frame_counts % 16)))
            gts = np.zeros(feature.shape[0] * 16)

            # @@@@@@TODO 토막 단위 또는 프레임 단위로 gt 만들기 @@@@@@@@@@@@@@@@@@@@

            gts[int(frames_gt[0]) - 1 : min(int(frames_gt[1]), frame_counts)] = 1

            if len(frames_gt) == 4:
                gts[int(frames_gt[2]) - 1 : min(int(frames_gt[3]), frame_counts)] = 1

            # # for i in range(12):
            # #     gts[180 + i] = gts[179]
            # # @@ feature extraction할때 마지막 조각에서 frame 개수가 16개가 안되면 마지막 frame을 복사해서 추가함

            gts = gts.reshape(-1, 16)

            # assert feature.shape[0] == gts.shape[0]

            # gts = np.max(gts, axis=1)
            # 16프레임중 1개라도 1이 있으면 True로 취급

            gts = np.mean(gts, axis=1)
            # 마지막에 self.gt_thr 넘는 값만 True로 취급
            # 기본값 0.25

            gts_npy = np.zeros(self.num_segments).astype(np.float32)

            for i in range(len(sample_index) - 1):
                if sample_index[i] == sample_index[i + 1]:
                    gts_npy[i] = gts[sample_index[i]]
                else:
                    gts_npy[i] = gts[sample_index[i] : sample_index[i + 1]].mean(0)
                    # ex의 0과 6 => [0:6] => 0~5 gts 6개 평균

            gts_npy = gts_npy > self.gt_thr
            gts_npy = gts_npy.astype(np.float32)

            return torch.from_numpy(feature_npy), torch.from_numpy(gts_npy)
        else:
            return torch.from_numpy(feature_npy)
