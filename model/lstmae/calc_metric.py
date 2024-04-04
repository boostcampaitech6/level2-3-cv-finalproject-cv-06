import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataset import AbnormalDataset, NormalDataset
from lstm_ae import LSTMAutoEncoder
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--abnormal_root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_CSV",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/abnormal",
        ),
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_VAL_JSON",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/label",
        ),
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get(
            "SM_MODEL_DIR",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pts",
        ),
    )

    parser.add_argument("--model_name", type=str, default="LSTM")
    parser.add_argument("--pth_name", type=str, default="LSTM_20240324_222238_best")
    parser.add_argument("--seed", type=int, default=666)

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--thr", type=float, default=0.02)

    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_run_name", type=str, default="LSTM_auc")

    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_image_to_wandb(image_path):
    with open(image_path, "rb") as file:
        img = Image.open(file)
        wandb.log(
            {
                image_path.split("/")[-1].split(".")[0]: [
                    wandb.Image(img, caption=f"{image_path.split('/')[-1]}")
                ]
            }
        )


def calculate_mse(seq1, seq2):
    return np.mean(np.power(seq1 - seq2, 2))


def train(
    abnormal_root_dir,
    json_dir,
    model_dir,
    model_name,
    pth_name,
    device,
    num_workers,
    batch_size,
    thr,
    seed,
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # Define parameters
    n_features = 38  # Number of features to predict
    sequence_length = 20

    batch_size = batch_size

    abnormal_dataset = AbnormalDataset(
        root=abnormal_root_dir,
        label_root=json_dir,
    )

    abnormal_loader = DataLoader(
        dataset=abnormal_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> data_load_time: {data_load_time}")

    # Initialize the LSTM autoencoder model
    model = LSTMAutoEncoder(
        num_layers=2, hidden_size=50, n_features=n_features, device=device
    )

    load_dict = torch.load(osp.join(model_dir, f"{pth_name}.pth"), map_location="cpu")

    model.load_state_dict(load_dict["model_state_dict"])
    model.to(device)

    val_criterion = nn.MSELoss(reduction="none")

    print(f"Start calculation auc..")

    wandb.init(
        project="VAD",
        config={
            "dataset": "무인매장",
            "loss": "MSE",
            "notes": "LSTM auc 구하기",
        },
        name=wandb_run_name + "_" + train_start,
        mode=wandb_mode,
    )

    wandb.watch((model,))
    model.eval()

    label_list = []
    mse_list = []
    pred_list = []

    with torch.no_grad():
        for step, (data, label) in tqdm(
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

            pred_list.append(1 if mse > thr else 0)

    conf_matrix_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/confusion_matrix.png"
    roc_curve_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/roc_curve.png"
    pr_curve_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/pr_curve.png"

    conf_matrix = confusion_matrix(label_list, pred_list)
    plt.figure(figsize=(7, 7))
    sns.heatmap(
        conf_matrix,
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
        annot=True,
        fmt="d",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(conf_matrix_path)

    false_pos_rate, true_pos_rate, thresholds = roc_curve(label_list, mse_list)
    roc_auc = auc(
        false_pos_rate,
        true_pos_rate,
    )

    plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label="AUC = %0.3f" % roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=5)

    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc="lower right")
    plt.title("ROC curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(roc_curve_path)

    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    acc = sum((pred_list == label_list)) / len(pred_list)
    print("accuracy 점수: {}".format(acc))
    print("roc_auc 점수: {}".format(roc_auc))

    new_wandb_metric_dict = {
        "thr": thr,
        "auc": roc_auc,
        "accuracy": acc,
    }

    wandb.log(new_wandb_metric_dict)

    save_image_to_wandb(conf_matrix_path)
    save_image_to_wandb(roc_curve_path)
    save_image_to_wandb(pr_curve_path)

    os.remove(conf_matrix_path)
    os.remove(roc_curve_path)
    os.remove(pr_curve_path)


def main(args):
    train(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
