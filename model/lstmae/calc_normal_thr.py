import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import NormalDataset
from lstm_ae import LSTMAutoEncoder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

"""
MSE loss
평균과 공분산 이용
# 평균[0.000071]
# 중간[0.000050]
# 최소[0.000002]
# 최대[0.001509]

loss 그대로 이용
# 평균[0.026619]
# 중간[0.026682]
# 최소[0.022427]
# 최대[0.030678]

MAE loss
# 평균[0.002338]
# 중간[0.001776]
# 최소[0.000088]
# 최대[0.025161]

=> MSE 는 너무 작아서 MAE 로 실시
"""


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(666)


def main():
    root_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/normal"
    dataset = NormalDataset(
        root=root_dir,
    )
    valid_data_size = len(dataset) // 10
    train_data_size = len(dataset) - valid_data_size

    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_data_size, valid_data_size]
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=8
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAutoEncoder(num_layers=2, hidden_size=50, n_features=38, device=device)
    load_dict = torch.load(
        "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pts/LSTM_20240324_222238_best.pth",
        map_location="cpu",
    )

    model.load_state_dict(load_dict["model_state_dict"])
    model.to(device)
    val_criterion = nn.MSELoss(reduction="none")

    loss_list = []
    loss_m_list = []
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for i, batch_data in tqdm(enumerate(valid_loader), total=len(valid_loader)):

            batch_data = batch_data.to(device)
            predict_values = model(batch_data)

            loss = val_criterion(predict_values, batch_data)
            loss_m = torch.mean(loss)
            loss_m_list.append(loss_m.cpu().numpy())

            loss_mae = F.l1_loss(predict_values, batch_data, reduction="none")
            loss_mae = loss_mae.mean()
            loss_list.append(loss_mae.cpu().numpy())

            total_loss += loss_m

    loss_list = np.array(loss_list)
    loss_m_list = np.array(loss_m_list)

    val_abnormal_mean_loss = (total_loss / len(valid_loader)).item()

    ## 정상구간에서 mse 점수 분포
    print("mae")
    print(
        "평균[{:.6f}]\n중간[{:.6f}]\n최소[{:.6f}]\n최대[{:.6f}]".format(
            np.mean(loss_list),
            np.median(loss_list),
            np.min(loss_list),
            np.max(loss_list),
        )
    )
    print("=" * 40)
    print("mse")
    print(
        "평균[{:.6f}]\n중간[{:.6f}]\n최소[{:.6f}]\n최대[{:.6f}]".format(
            np.mean(loss_m_list),
            np.median(loss_m_list),
            np.min(loss_m_list),
            np.max(loss_m_list),
        )
    )
    print("=" * 40)
    print("total_loss: {}".format(val_abnormal_mean_loss))


if __name__ == "__main__":

    main()
