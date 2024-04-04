import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import AbnormalDataset
from lstm_ae import LSTMAutoEncoder
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
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


def calculate_mse(seq1, seq2):
    return np.mean(np.power(seq1 - seq2, 2))


def main():
    root_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/abnormal"
    json_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/label"
    abnormal_dataset = AbnormalDataset(
        root=root_dir,
        label_root=json_dir,
    )
    abnormal_loader = DataLoader(
        dataset=abnormal_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    sequence_length = 20
    n_features = 38

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAutoEncoder(num_layers=2, hidden_size=50, n_features=38, device=device)
    load_dict = torch.load(
        "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pts/LSTM_20240324_222238_best.pth",
        map_location="cpu",
    )

    model.load_state_dict(load_dict["model_state_dict"])
    model.to(device)
    val_criterion = nn.MSELoss(reduction="none")

    label_list = []
    mse_list = []

    model.eval()

    with torch.no_grad():

        for i, (data, label) in tqdm(
            enumerate(abnormal_loader), total=len(abnormal_loader)
        ):
            scaler = MinMaxScaler()

            label = label.reshape(-1).cpu().numpy()

            if sum(label) >= 1:
                label_list.append(1)
            else:
                label_list.append(0)

            data = data.cpu().detach().numpy()
            data = data.reshape(sequence_length, n_features)
            data = scaler.fit_transform(data)
            scaled_data = data.reshape(1, sequence_length, n_features)
            scaled_data = torch.from_numpy(scaled_data).float().to(device)

            pred = model(scaled_data)
            pred = pred.cpu().detach().numpy().reshape(-1, n_features)
            # pred_original = scaler.inverse_transform(pred.cpu().detach().numpy().reshape(-1, n_features))

            mse = calculate_mse(data, pred)
            mse_list.append(mse)

    precision_rt, recall_rt, threshold_rt = precision_recall_curve(label_list, mse_list)

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_rt, precision_rt[1:], label="Precision")
    plt.plot(threshold_rt, recall_rt[1:], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Precision/Recall")
    plt.legend()
    plt.savefig(
        "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/pr_curve.png"
    )

    # best position of threshold
    index_cnt = [
        cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p == r
    ][0]
    print("precision: ", precision_rt[index_cnt], ", recall: ", recall_rt[index_cnt])

    # fixed Threshold
    threshold_fixed = threshold_rt[index_cnt]
    print("threshold: ", threshold_fixed)

    print("mse mean: ", np.mean(mse))


if __name__ == "__main__":

    main()

# 이상행동 20 중 5프레임 이상
# precision:  0.591683741111245 , recall:  0.591683741111245
# threshold:  0.02313113

# 이상행동 20 중 1프레임 이상
# precision:  0.6168715461824743 , recall:  0.6168715461824743
# threshold:  0.02235273

# 이상행동 20 중 20프레임 모두
# precision:  0.4549022511848341 , recall:  0.4549022511848341
# threshold:  0.02781844

# 이상행동 20 중 3프레임 이상
# precision:  0.6046858260748056 , recall:  0.6046858260748056
# threshold:  0.022715755

# min_max_scaler inverse 후

# 이상행동 20 중 1프레임 이상
# precision:  0.46159918800045113 , recall:  0.46159918800045113
# threshold:  0.0026742023210807914

# 이상행동 20 중 10프레임 이상
# precision:  0.37000934704232874 , recall:  0.37000934704232874
# threshold:  0.003047690912489323

# 이상행동 20 중 3프레임 이상
# precision:  0.44426848013414694 , recall:  0.44426848013414694
# threshold:  0.002750229995391417
