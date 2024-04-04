import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from classifier import LSTMAutoencoder
from shop_dataset import AbnormalDataset, NormalDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_TRAIN_CSV",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv/normal/val",
        ),
    )
    # 학습 데이터 경로
    parser.add_argument(
        "--abnormal_root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_CSV",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv/abnormal/val",
        ),
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_VAL_JSON",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal/val",
        ),
    )
    # abnormal 검증셋 csv, json파일 경로
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/data/ephemeral/home/pths"),
    )
    # pth 파일 저장 경로

    parser.add_argument("--model_name", type=str, default="LSTM")
    # import_module로 불러올 model name

    parser.add_argument("--resume_name", type=str, default="")
    # resume 파일 이름

    parser.add_argument("--seed", type=int, default=666)
    # random seed

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--val_num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=50)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--thr", type=float, default=0.02)

    parser.add_argument("--patience", type=int, default=10)

    # parser.add_argument("--mp", action="store_false")
    # https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    # mixed precision 사용할 지 여부

    parser.add_argument("--wandb_mode", type=str, default="online")
    # parser.add_argument("--wandb_mode", type=str, default="disabled")
    # wandb mode
    parser.add_argument("--wandb_run_name", type=str, default="LSTM")
    # wandb run name

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


def train(
    root_dir,
    abnormal_root_dir,
    json_dir,
    model_dir,
    model_name,
    device,
    num_workers,
    batch_size,
    val_num_workers,
    val_batch_size,
    learning_rate,
    max_epoch,
    val_interval,
    save_interval,
    thr,
    patience,
    resume_name,
    seed,
    # mp,
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # Define parameters
    sequence_length = 20  # Adjust as needed
    prediction_time = 1  # Adjust as needed
    n_features = 38  # Number of features to predict

    batch_size = batch_size
    val_batch_size = val_batch_size

    # -- early stopping flag
    patience = patience
    counter = 0

    # 데이터셋
    dataset = NormalDataset(
        root=root_dir,
    )

    valid_data_size = len(dataset) // 10

    train_data_size = len(dataset) - valid_data_size

    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_data_size, valid_data_size]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
    )

    abnormal_dataset = AbnormalDataset(
        root=abnormal_root_dir,
        label_root=json_dir,
    )

    abnormal_loader = DataLoader(
        dataset=abnormal_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=val_num_workers,
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> data_load_time: {data_load_time}")

    # Initialize the LSTM autoencoder model
    model = LSTMAutoencoder(sequence_length, n_features, prediction_time)

    # load_dict = None

    # if resume_name:
    #     load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
    #     model.load_state_dict(load_dict["model_state_dict"])

    # model.load_state_dict(
    #     torch.load(
    #         "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pytorch_model.pth",
    #         map_location="cpu",
    #     )
    # )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 40], gamma=0.1
    )

    # if resume_name:
    #     optimizer.load_state_dict(load_dict["optimizer_state_dict"])
    #     scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.MSELoss()
    val_criterion = nn.MSELoss(reduction="none")

    print(f"Start training..")

    wandb.init(
        project="VAD",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "무인매장",
            "n_epochs": max_epoch,
            "loss": "MSE",
            "notes": "VAD 실험",
        },
        name=wandb_run_name + "_" + train_start,
        mode=wandb_mode,
    )

    wandb.watch((model,))

    best_loss = np.inf

    total_batches = len(train_loader)

    for epoch in range(max_epoch):
        model.train()

        epoch_start = datetime.now()

        epoch_loss = 0

        for step, (x, y, _) in tqdm(enumerate(train_loader), total=total_batches):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss

        epoch_mean_loss = (epoch_loss / total_batches).item()

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(
            f"==>> epoch {epoch+1} train_time: {train_time}\nloss: {round(epoch_mean_loss,4)}"
        )

        if (epoch + 1) % save_interval == 0:

            ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_latest.pth")

            states = {
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                # "scaler_state_dict": scaler.state_dict(),
            }

            torch.save(states, ckpt_fpath)

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % val_interval == 0:

            print(f"Start validation #{epoch+1:2d}")
            model.eval()

            with torch.no_grad():
                total_loss = 0
                total_abnormal_loss = 0
                total_n_corrects = 0
                total_abnormal_n_corrects = 0
                total_auc = 0
                total_abnormal_auc = 0
                error_count = 0
                error_count_abnormal = 0

                for step, (x, y, label) in tqdm(
                    enumerate(valid_loader), total=len(valid_loader)
                ):
                    x, y, label = x.to(device), y.to(device), label.to(device)

                    pred = model(x)

                    val_loss = val_criterion(pred, y)
                    val_loss_rdim = torch.mean(val_loss, dim=2)
                    pred_label = val_loss_rdim > thr
                    # pred_sig = F.sigmoid(val_loss_rdim-thr)
                    label = label.view(-1, 1)

                    try:
                        auc = roc_auc_score(label.cpu(), pred_label.cpu())
                        # auc = roc_auc_score(label.cpu(), pred_sig.cpu())
                        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                        # 정상상황인 경우 label이 항상 전부 0
                        # => 무조건 "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case." 발생
                        total_auc += auc
                    except ValueError:
                        # print(
                        #     "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case."
                        # )
                        total_auc += 0
                        error_count += 1

                    pred_correct = pred_label == label
                    corrects = torch.sum(pred_correct).item()

                    total_n_corrects += corrects

                    val_loss = torch.mean(val_loss)

                    total_loss += val_loss

                val_mean_loss = (total_loss / len(valid_loader)).item()
                if error_count < len(valid_loader):
                    val_auc = total_auc / (len(valid_loader) - error_count)
                else:
                    # 정상영상은 roc_auc_score 함수 사용 불가 => error_count == len(valid_loader)
                    val_auc = 0
                    # ==> vaild_auc는 항상 0
                val_accuracy = total_n_corrects / valid_data_size

                for step, (x, y, label) in tqdm(
                    enumerate(abnormal_loader), total=len(abnormal_loader)
                ):
                    x, y, label = x.to(device), y.to(device), label.to(device)

                    pred = model(x)

                    val_loss = val_criterion(pred, y)
                    val_loss_rdim = torch.mean(val_loss, dim=2)
                    pred_label = val_loss_rdim > thr
                    # pred_sig = F.sigmoid(val_loss_rdim - thr)
                    label = label.view(-1, 1)

                    try:
                        auc = roc_auc_score(label.cpu(), pred_label.cpu())
                        # auc = roc_auc_score(label.cpu(), pred_sig.cpu())
                        total_abnormal_auc += auc
                    except ValueError:
                        # print(
                        #     "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case."
                        # )
                        total_abnormal_auc += 0
                        error_count_abnormal += 1

                    pred_correct = pred_label == label
                    corrects = torch.sum(pred_correct).item()

                    total_abnormal_n_corrects += corrects

                    val_loss = torch.mean(val_loss)

                    total_abnormal_loss += val_loss

                val_abnormal_mean_loss = (
                    total_abnormal_loss / len(abnormal_loader)
                ).item()
                val_abnormal_auc = total_abnormal_auc / (
                    len(abnormal_loader) - error_count_abnormal
                )
                val_abnormal_accuracy = total_abnormal_n_corrects / len(
                    abnormal_dataset
                )

                val_total_auc = (total_auc + total_abnormal_auc) / (
                    len(valid_loader)
                    + len(abnormal_loader)
                    - error_count
                    - error_count_abnormal
                )
                val_total_accuracy = (total_n_corrects + total_abnormal_n_corrects) / (
                    valid_data_size + len(abnormal_dataset)
                )

            if best_loss > val_mean_loss:
                print(
                    f"Best performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_loss:.4f}"
                )
                print(f"Save model in {model_dir}")
                states = {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                    # "optimizer_state_dict": optimizer.state_dict(),
                    # "scheduler_state_dict": scheduler.state_dict(),
                    # "scaler_state_dict": scaler.state_dict(),
                    # best.pth는 inference에서만 쓰기?
                }

                best_ckpt_fpath = osp.join(
                    model_dir, f"{model_name}_{train_start}_best.pth"
                )
                torch.save(states, best_ckpt_fpath)
                best_loss = val_mean_loss
                counter = 0
            else:
                counter += 1

        new_wandb_metric_dict = {
            "train_loss": epoch_mean_loss,
            "valid_loss": val_mean_loss,
            "valid_abnormal_loss": val_abnormal_mean_loss,
            "valid_auc": val_auc,
            "valid_abnormal_auc": val_abnormal_auc,
            "valid_normal+abnormal_auc": val_total_auc,
            "valid_accuracy": val_accuracy,
            "valid_abnormal_accuracy": val_abnormal_accuracy,
            "valid_normal+abnormal_accuracy": val_total_accuracy,
            "learning_rate": scheduler.get_lr()[0],
        }

        wandb.log(new_wandb_metric_dict)

        scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print(
            f"==>> epoch {epoch+1} time: {epoch_time}\nvalid_loss: {round(val_mean_loss,4)}\nvalid_auc: {val_auc:.4f}\nvalid_accuracy: {val_accuracy:.2f}"
        )
        print(
            f"valid_abnormal_loss: {round(val_abnormal_mean_loss,4)}\nvalid_abnormal_auc: {val_abnormal_auc:.4f}\nvalid_abnormal_accuracy: {val_abnormal_accuracy:.2f}"
        )
        print(
            f"valid_normal+abnormal_auc: {val_total_auc:.4f}\nvalid_normal+abnormal_accuracy: {val_total_accuracy:.2f}"
        )
        print(f"auc_roc_error_count: {error_count+error_count_abnormal}")

        if counter > patience:
            print("Early Stopping...")
            break

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(f"==>> total time: {total_time}")


def main(args):
    train(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()

    main(args)
