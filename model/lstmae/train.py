import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataset import NormalDataset
from lstm_ae import LSTMAutoEncoder
from lstm_ae_old import LSTMAutoencoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()

    # 학습 데이터 경로
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_TRAIN_CSV",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/lstmae/dataset/normal",
        ),
    )

    # pth 파일 저장 경로
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get(
            "SM_MODEL_DIR",
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pts",
        ),
    )

    # import_module로 불러올 model name
    parser.add_argument("--model_name", type=str, default="LSTM")
    # resume 파일 이름
    parser.add_argument("--resume_name", type=str, default="")
    # random seed
    parser.add_argument("--seed", type=int, default=666)

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

    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_run_name", type=str, default="LSTM-AE")

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
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # Define parameters
    sequence_length = 20
    prediction_time = 1
    n_features = 38

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

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> data_load_time: {data_load_time}")

    # Initialize the LSTM autoencoder model
    # model = LSTMAutoencoder(sequence_length, prediction_time, n_features, 50)
    # model.to(device)

    model = LSTMAutoEncoder(
        num_layers=2, hidden_size=50, n_features=n_features, device=device
    )
    model.to(device)

    optimizer = torch.optim.Adam(
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

        for step, data in tqdm(enumerate(train_loader), total=total_batches):

            data = data.to(device)
            optimizer.zero_grad()

            pred = model(data)

            loss = criterion(pred, data)

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

                for step, data in tqdm(
                    enumerate(valid_loader), total=len(valid_loader)
                ):

                    data = data.to(device)

                    pred = model(data)

                    val_loss = val_criterion(pred, data)
                    val_loss = torch.mean(val_loss)

                    total_loss += val_loss

                val_mean_loss = (total_loss / len(valid_loader)).item()

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
            "learning_rate": scheduler.get_lr()[0],
        }

        wandb.log(new_wandb_metric_dict)

        scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print(
            f"==>> epoch {epoch+1} time: {epoch_time}\nvalid_loss: {round(val_mean_loss,4)}"
        )

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
