import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler

from argparse import ArgumentParser

import os
import os.path as osp

from tqdm import tqdm

import wandb

from shop_dataset import NormalVMAE, AbnormalVMAE
from classifier import MILClassifier, MILandFE
from loss import MIL


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--normal_root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_NORMAL_NPY",
            "../datapreprocess/npy/normal",
        ),
    )
    # 학습 데이터 경로
    parser.add_argument(
        "--abnormal_root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_NPY",
            "../datapreprocess/npy/abnormal",
        ),
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_JSON",
            "../datapreprocess/json/abnormal",
        ),
    )
    # abnormal 검증셋 npy, json파일 경로
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "../pths"))
    # pth 파일 저장 경로

    parser.add_argument("--model_name", type=str, default="MIL")
    # import_module로 불러올 model name

    parser.add_argument("--resume_name", type=str, default="")
    # resume 파일 이름

    parser.add_argument("--model_size", type=str, default="small")
    # VideoMAEv2 backbone 사이즈 = "small" or "base"

    parser.add_argument("--seed", type=int, default=666)
    # random seed

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=30)
    # parser.add_argument("--val_batch_size", type=int, default=1)
    # parser.add_argument("--val_num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=1000)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--thr", type=float, default=0.25)
    parser.add_argument("--drop_rate", type=float, default=0.3)

    parser.add_argument("--patience", type=int, default=100)

    # parser.add_argument("--mp", action="store_false")
    # https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    # mixed precision 사용할 지 여부

    parser.add_argument("--use_extra", action="store_false")

    # parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    # wandb mode
    parser.add_argument("--wandb_run_name", type=str, default="MIL")
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
    normal_root_dir,
    abnormal_root_dir,
    json_dir,
    model_dir,
    model_name,
    model_size,
    device,
    num_workers,
    batch_size,
    # val_num_workers,
    # val_batch_size,
    learning_rate,
    max_epoch,
    val_interval,
    save_interval,
    thr,
    drop_rate,
    patience,
    resume_name,
    seed,
    # mp,
    use_extra,
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = batch_size

    val_batch_size = 1
    val_num_workers = 0

    # -- early stopping flag
    patience = patience
    counter = 0

    # 데이터셋
    dataset = NormalVMAE(
        model_size=model_size,
        root=normal_root_dir,
    )

    valid_data_size = len(dataset) // 10

    train_data_size = len(dataset) - valid_data_size

    train_dataset, valid_dataset = random_split(dataset, lengths=[train_data_size, valid_data_size])

    normal_train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )

    # normal_valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=val_batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     num_workers=val_num_workers,
    # )

    abnormal_train_dataset = AbnormalVMAE(
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=json_dir,
    )
    abnormal_valid_dataset = AbnormalVMAE(
        is_train=0,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=json_dir,
    )

    abnormal_train_loader = DataLoader(
        dataset=abnormal_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # abnormal_valid_loader = DataLoader(
    #     dataset=abnormal_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    # )

    concat_valid_dataset = ConcatDataset([valid_dataset, abnormal_valid_dataset])
    concat_valid_loader = DataLoader(
        dataset=concat_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> {model_size} data_load_time: {data_load_time}")

    # Initialize the model
    # model = MILClassifier(drop_p=drop_rate)
    model = MILandFE(drop_p=drop_rate)

    load_dict = None

    if resume_name:
        load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
        model.load_state_dict(load_dict["model_state_dict"])

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # 1e-6 => 0.0010000000474974513
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)

    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.BCELoss()
    MIL_criterion = MIL

    print(f"Start training..")

    wandb.init(
        project="VAD",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "무인매장",
            "n_epochs": max_epoch,
            "loss": "BCE+MIL",
            "notes": "VAD 실험",
        },
        name=wandb_run_name + "_" + train_start,
        mode=wandb_mode,
    )

    wandb.watch((model,))

    best_loss = np.inf
    best_auc = 0

    total_batches = len(abnormal_train_loader)

    for epoch in range(max_epoch):
        model.train()

        epoch_start = datetime.now()

        epoch_loss = 0
        epoch_n_corrects = 0
        epoch_n_MIL_loss = 0
        epoch_n_loss = 0
        epoch_n_n_corrects = 0

        epoch_abnormal_max = 0
        epoch_abnormal_mean = 0
        epoch_normal_max = 0
        epoch_normal_mean = 0

        norm_train_iter = iter(normal_train_loader)
        # iterator를 여기서 매번 새로 할당해줘야 iterator가 다시 처음부터 작동

        for step, abnormal_inputs in tqdm(
            enumerate(abnormal_train_loader),
            total=total_batches,
        ):
            try:
                normal_inputs = next(norm_train_iter)

                abnormal_input, abnormal_gt = abnormal_inputs
                # (batch_size, 12, 710), (batch_size, 12)
                normal_input, normal_gt = normal_inputs
                # (batch_size, 12, 710), (batch_size, 12)

                inputs, gts = torch.cat((abnormal_input, normal_input), dim=1), torch.cat(
                    (abnormal_gt, normal_gt), dim=1
                )
                # inputs는 (batch_size, 24, 710), gts는 (batch_size, 24)

                # batch_size = inputs.shape[0]

                # inputs = inputs.view(-1, inputs.size(-1)).to(device)
                # (batch_size * 24, 710)
                inputs = inputs.to(device)
                gts = gts.view(-1, 1).to(device)
                # (batch_size * 24, 1)

                optimizer.zero_grad()

                pred = model(inputs)
                # pred는 (batch_size * 24, 1)

                loss = criterion(pred, gts)
                MIL_loss = MIL_criterion(pred, batch_size, abnormal_input.size(1))
                sum_loss = loss + MIL_loss
                # sum_loss = MIL_loss
                sum_loss.backward()

                # loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred_a = pred.view(batch_size, 2, abnormal_input.size(1))[:, 0, :]
                    pred_n = pred.view(batch_size, 2, abnormal_input.size(1))[:, 1, :]

                    pred_a_max = torch.mean(torch.max(pred_a, dim=-1)[0])
                    pred_n_max = torch.mean(torch.max(pred_n, dim=-1)[0])

                    pred_a_mean = torch.mean(pred_a)
                    pred_n_mean = torch.mean(pred_n)

                    pred_correct = pred > thr
                    gts_correct = gts > thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    epoch_n_loss += loss.item()
                    epoch_n_MIL_loss += MIL_loss.item()

                    epoch_n_n_corrects += corrects / (abnormal_input.size(1) * 2)

                    epoch_abnormal_max += pred_a_max.item()
                    epoch_abnormal_mean += pred_a_mean.item()
                    epoch_normal_max += pred_n_max.item()
                    epoch_normal_mean += pred_n_mean.item()

            except StopIteration:
                if not use_extra:

                    break
                abnormal_input, abnormal_gt = abnormal_inputs
                # (batch_size, 12, 710), (batch_size, 12)

                # inputs = abnormal_input.view(-1, inputs.size(-1)).to(device)
                # (batch_size * 12, 710)
                inputs = abnormal_input.to(device)
                gts = abnormal_gt.view(-1, 1).to(device)
                # (batch_size * 12, 1)

                optimizer.zero_grad()

                pred = model(inputs)
                # pred는 (batch_size * 12, 1)

                loss = criterion(pred, gts)

                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # print(f"==>> pred.shape: {pred.shape}")
                    pred_a = pred.view(batch_size, abnormal_input.size(1))

                    pred_a_max = torch.mean(torch.max(pred_a, dim=-1)[0])

                    pred_a_mean = torch.mean(pred_a)

                    pred_correct = pred > thr
                    gts_correct = gts > thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    epoch_loss += loss.item()
                    epoch_n_corrects += corrects / abnormal_input.size(1)

                    epoch_abnormal_max += pred_a_max.item()
                    epoch_abnormal_mean += pred_a_mean.item()

        epoch_n_mean_loss = epoch_n_loss / len(normal_train_loader)
        epoch_n_mean_MIL_loss = epoch_n_MIL_loss / len(normal_train_loader)
        epoch_n_accuracy = epoch_n_n_corrects / (batch_size * (len(normal_train_loader)))

        epoch_mean_normal_max = epoch_normal_max / len(normal_train_loader)
        epoch_mean_normal_mean = epoch_normal_mean / len(normal_train_loader)
        if use_extra:
            epoch_mean_loss = (epoch_loss + epoch_n_loss) / total_batches
            epoch_accuracy = (epoch_n_corrects + epoch_n_n_corrects) / (
                batch_size * (len(abnormal_train_loader))
            )
            epoch_mean_abnormal_max = epoch_abnormal_max / total_batches
            epoch_mean_abnormal_mean = epoch_abnormal_mean / total_batches
        else:
            epoch_mean_loss = (epoch_loss + epoch_n_loss) / len(normal_train_loader)
            epoch_accuracy = (epoch_n_corrects + epoch_n_n_corrects) / (
                batch_size * (len(normal_train_loader))
            )
            epoch_mean_abnormal_max = epoch_abnormal_max / len(normal_train_loader)
            epoch_mean_abnormal_mean = epoch_abnormal_mean / len(normal_train_loader)

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(
            f"==>> epoch {epoch+1} train_time: {train_time}\nloss: {round(epoch_mean_loss,4)} n_loss: {round(epoch_n_mean_loss,4)} MIL_loss: {round(epoch_n_mean_MIL_loss,4)}"
        )
        print(f"accuracy: {epoch_accuracy:.2f} n_accuracy: {epoch_n_accuracy:.2f}")
        print(f"==>> abnormal_max_mean: {epoch_mean_abnormal_max} abnormal_mean: {epoch_mean_abnormal_mean}")
        print(f"==>> normal_max_mean: {epoch_mean_normal_max} normal_mean: {epoch_mean_normal_mean}")

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

                total_n_corrects = 0

                total_max = 0
                total_mean = 0

                total_TP = 0
                total_FN = 0
                total_num_TRUE = 0
                total_FP = 0
                total_TN = 0
                total_num_FALSE = 0
                total_preds = []
                total_gts = []

                for step, (inputs, gts) in tqdm(
                    enumerate(concat_valid_loader), total=len(concat_valid_loader)
                ):
                    inputs = inputs.to(device)
                    # (val_batch_size, 11, 710)
                    # gts = gts.to(device)
                    # 이상영상이면 (val_batch_size, 176)
                    # 정상영상이면 (val_batch_size, 11)
                    if gts.size(1) == inputs.size(1):
                        gts11 = gts.view(-1, 1).to(device)
                        gts176 = torch.zeros(val_batch_size, gts.size(1) * 16)
                    else:
                        gts11 = (torch.max(gts.view(-1, inputs.size(1), 16), dim=2)[0]).view(-1, 1).to(device)
                        gts176 = gts

                    pred = model(inputs)
                    # pred는 (val_batch_size * 11, 1)

                    val_loss = criterion(pred, gts11)

                    pred_view = pred.view(val_batch_size, inputs.size(1))

                    pred_max = torch.mean(torch.max(pred_view, dim=-1)[0])

                    pred_mean = torch.mean(pred_view)

                    pred_correct = pred > thr
                    gts_correct = gts11 > thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    pred = (pred.squeeze()).detach().cpu().numpy()
                    pred_np = np.zeros(gts176.size(1))

                    step = np.array([i for i in range(gts176.size(1) + 1)])

                    for j in range(inputs.size(1)):
                        pred_np[step[j] * 16 : step[j + 1] * 16] = pred[j]

                    gts176 = gts176.squeeze().detach().numpy()

                    pred_positive = pred_np > thr
                    TP_and_FN = pred_positive[gts176 > 0.9]
                    FP_and_TN = pred_positive[gts176 < 0.1]

                    num_TP = np.sum(TP_and_FN)
                    num_FP = np.sum(FP_and_TN)

                    total_TP += num_TP
                    total_FN += len(TP_and_FN) - num_TP
                    total_FP += num_FP
                    total_TN += len(FP_and_TN) - num_FP

                    total_preds.append(pred_np)
                    total_gts.append(gts176)

                    total_num_TRUE += len(TP_and_FN)
                    total_num_FALSE += len(FP_and_TN)
                    total_n_corrects += corrects / inputs.size(1)

                    total_loss += val_loss.item()

                    total_max += pred_max.item()
                    total_mean += pred_mean.item()

                val_mean_loss = total_loss / len(concat_valid_loader)

                val_tpr = total_TP / total_num_TRUE
                val_fpr = total_FP / total_num_FALSE

                val_accuracy = total_n_corrects / len(concat_valid_loader)
                # for loop 한번에 abnormal 12, normal 12해서 24개 정답 확인

                val_mean_max = total_max / len(concat_valid_loader)
                val_mean_mean = total_mean / len(concat_valid_loader)

            if epoch == 0 or (epoch + 1) % (100 * val_interval) == 0:
                conf_mtx = np.array([[total_TP, total_FN], [total_FP, total_TN]])
                fig = plt.figure(figsize=(14, 7))
                ax = fig.add_subplot(121, aspect=1)
                sns.heatmap(
                    conf_mtx,
                    xticklabels=["Abnormal", "Normal"],
                    yticklabels=["Abnormal", "Normal"],
                    annot=True,
                    fmt="d",
                    ax=ax,
                )
                ax.set_title("Confusion matrix")
                ax.set_xlabel("Preds")
                ax.set_ylabel("GTs")

                ax2 = fig.add_subplot(122, aspect=1)
                total_preds_np = np.concatenate(total_preds)
                total_gts_np = np.concatenate(total_gts)

                fpr, tpr, cut = roc_curve(y_true=total_gts_np, y_score=total_preds_np)

                diff = tpr - fpr
                diff_idx = np.argmax(diff)
                best_thr = cut[diff_idx]
                val_bthr = best_thr if diff_idx != 0 else 1
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

                ax2.plot(fpr, tpr, linewidth=5, label=f"AUC = {val_auc}")
                ax2.plot([0, 1], [0, 1], linewidth=5)

                ax2.set_xlim([-0.01, 1])
                ax2.set_ylim([0, 1.01])
                ax2.legend(loc="lower right")
                ax2.set_title("ROC curve")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_xlabel("False Positive Rate")

                wandb.log({f"Confusion matrix and ROC": wandb.Image(fig, caption=f"{epoch+1} epoch")})
                # https://stackoverflow.com/questions/72134168/how-does-one-save-a-plot-in-wandb-with-wandb-log

                plt.clf()
                plt.close()

                # if epoch + 1 == max_epoch:
                print("Drawing graphs...")
                for i, (pred_g, gt_g) in tqdm(enumerate(zip(total_preds, total_gts)), total=len(total_preds)):
                    fig = plt.figure(figsize=(7, 7))
                    ax = fig.add_subplot(111)
                    ax.set_ylim([0, 1.01])
                    ax.set_title(f"Video {i} score graph")
                    ax.set_ylabel("Pred score")
                    ax.set_xlabel("Segment")
                    lins = np.linspace(0, len(pred_g) - 1, len(pred_g))

                    checker = np.concatenate((gt_g[1:], [0]))

                    gt_indices = np.where(gt_g != checker)[0]

                    for idx in range(len(gt_indices) // 2):

                        ax.axvspan(
                            gt_indices[2 * idx] + 0.5,
                            gt_indices[2 * idx + 1] + 0.5,
                            color="gray",
                            linestyle="--",
                            zorder=0,
                            alpha=0.3,
                        )

                    ax.plot(lins, pred_g)

                    if not os.path.exists(f"./graphs/{wandb_run_name}/{epoch}/"):
                        os.makedirs(f"./graphs/{wandb_run_name}/{epoch}/")

                    plt.savefig(f"./graphs/{wandb_run_name}/{epoch}/{i}.jpg", format="jpeg")

                    plt.clf()
                    plt.close()
            else:
                total_preds_np = np.concatenate(total_preds)
                total_gts_np = np.concatenate(total_gts)

                fpr, tpr, cut = roc_curve(y_true=total_gts_np, y_score=total_preds_np)

                diff = tpr - fpr
                diff_idx = np.argmax(diff)
                best_thr = cut[diff_idx]
                val_bthr = best_thr if diff_idx != 0 else 1
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

            if best_loss > val_mean_loss:
                str_to_keep = f"Best auc performance at epoch: {epoch + 1}, {best_auc:.4f} -> {val_auc:.4f}"
                print(str_to_keep)
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

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best.pth")
                torch.save(states, best_ckpt_fpath)
                best_loss = val_mean_loss
                # counter = 0
            # else:
            #     counter += 1

            if best_auc < val_auc:
                print(f"Best auc performance at epoch: {epoch + 1}, {best_auc:.4f} -> {val_auc:.4f}")
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

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best_auc.pth")
                torch.save(states, best_ckpt_fpath)
                best_auc = val_auc
                counter = 0
            else:
                counter += 1

        new_wandb_metric_dict = {
            "train_loss": epoch_mean_loss,
            "train_accuracy": epoch_accuracy,
            "train_n_loss": epoch_n_mean_loss,
            "train_n_MIL_loss": epoch_n_mean_MIL_loss,
            "train_n_accuracy": epoch_n_accuracy,
            "valid_loss": val_mean_loss,
            "valid_fpr": val_fpr,
            "valid_tpr": val_tpr,
            "valid_bthr": val_bthr,
            "valid_auc": val_auc,
            "valid_ap": val_ap,
            "valid_accuracy": val_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_abnormal_max_mean": epoch_mean_abnormal_max,
            "train_abnormal_mean": epoch_mean_abnormal_mean,
            "train_normal_max_mean": epoch_mean_normal_max,
            "train_normal_mean": epoch_mean_normal_mean,
            "valid_max_mean": val_mean_max,
            "valid_mean": val_mean_mean,
        }

        wandb.log(new_wandb_metric_dict)

        scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print(f"==>> epoch {epoch+1} time: {epoch_time}\nvalid_loss: {round(val_mean_loss,4)}")
        print(f"valid_tpr: {val_tpr} valid_fpr: {val_fpr} valid_bthr: {val_bthr}")
        print(f"valid_auc: {val_auc:.4f} valid_ap: {val_ap:.4f} valid_accuracy: {val_accuracy:.2f}")
        print(f"==>> val_max_mean: {val_mean_max} val_mean: {val_mean_mean}")

        if counter > patience:
            print("Early Stopping...")
            break

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(str_to_keep)
    print(f"==>> total time: {total_time}")


def train2(
    normal_root_dir,
    abnormal_root_dir,
    json_dir,
    model_dir,
    model_name,
    model_size,
    device,
    num_workers,
    batch_size,
    # val_num_workers,
    # val_batch_size,
    learning_rate,
    max_epoch,
    val_interval,
    save_interval,
    thr,
    drop_rate,
    patience,
    resume_name,
    seed,
    # mp,
    use_extra,
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = batch_size

    val_batch_size = 1
    val_num_workers = 0

    # -- early stopping flag
    patience = patience
    counter = 0

    # 데이터셋
    dataset = NormalVMAE(
        model_size=model_size,
        root=normal_root_dir,
    )

    valid_data_size = len(dataset) // 10

    train_data_size = len(dataset) - valid_data_size

    train_dataset, valid_dataset = random_split(dataset, lengths=[train_data_size, valid_data_size])

    # normal_train_loader = DataLoader(
    #     dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    # )

    # normal_valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=val_batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     num_workers=val_num_workers,
    # )

    abnormal_train_dataset = AbnormalVMAE(
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=json_dir,
    )
    abnormal_valid_dataset = AbnormalVMAE(
        is_train=0,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=json_dir,
    )

    # abnormal_train_loader = DataLoader(
    #     dataset=abnormal_train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=num_workers,
    # )

    concat_trainset = ConcatDataset([train_dataset, abnormal_train_dataset])

    concat_train_loader = DataLoader(
        dataset=concat_trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )

    # abnormal_valid_loader = DataLoader(
    #     dataset=abnormal_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    # )

    concat_valid_dataset = ConcatDataset([valid_dataset, abnormal_valid_dataset])
    concat_valid_loader = DataLoader(
        dataset=concat_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> {model_size} data_load_time: {data_load_time}")

    # Initialize the LSTM autoencoder model
    model = MILClassifier(drop_p=drop_rate)

    load_dict = None

    if resume_name:
        load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
        model.load_state_dict(load_dict["model_state_dict"])

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # 1e-6 => 0.0010000000474974513
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)

    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.BCELoss()
    MIL_criterion = MIL

    print(f"Start training..")

    wandb.init(
        project="VAD",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "무인매장",
            "n_epochs": max_epoch,
            "loss": "BCE",
            "notes": "VAD 실험",
        },
        name=wandb_run_name + "_" + train_start,
        mode=wandb_mode,
    )

    wandb.watch((model,))

    best_loss = np.inf
    best_auc = 0

    total_batches = len(concat_train_loader)

    for epoch in range(max_epoch):
        model.train()

        epoch_start = datetime.now()

        epoch_loss = 0
        epoch_n_corrects = 0

        epoch_abnormal_max = 0
        epoch_abnormal_mean = 0
        epoch_normal_max = 0
        epoch_normal_mean = 0

        nan_count = 0

        for step, inputs in tqdm(
            enumerate(concat_train_loader),
            total=total_batches,
        ):
            inp, gts = inputs
            # (batch_size, 11, 710), (batch_size, 11)

            num_segs = inp.size(1)

            # inp = inp.view(-1, inp.size(-1)).to(device)
            # (batch_size * 11, 710)
            inp = inp.to(device)
            gts = gts.view(-1, 1).to(device)
            # (batch_size * 11, 1)

            optimizer.zero_grad()

            pred = model(inp)
            # pred는 (batch_size * 11, 1)

            loss = criterion(pred, gts)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # print(f"==>> pred.shape: {pred.shape}")

                pred_correct = pred > thr
                gts_correct = gts > thr

                pred_correct = pred_correct == gts_correct
                corrects = torch.sum(pred_correct).item()

                epoch_loss += loss.item()
                epoch_n_corrects += corrects / (num_segs * batch_size)

                check = gts.view(batch_size, num_segs) != 0
                check = torch.sum(check, dim=1)

                check_a = check != 0
                check_n = check_a == False

                pred_reshape = pred.view(batch_size, num_segs)

                pred_a_max = torch.mean(torch.max(pred_reshape[check_a], dim=-1)[0])

                pred_a_mean = torch.mean(pred_reshape[check_a])

                epoch_abnormal_max += pred_a_max.item()
                epoch_abnormal_mean += pred_a_mean.item()

                if torch.sum(check_n) != 0:
                    pred_n_max = torch.mean(torch.max(pred_reshape[check_n], dim=-1)[0])

                    pred_n_mean = torch.mean(pred_reshape[check_n])

                    epoch_normal_max += pred_n_max.item()
                    epoch_normal_mean += pred_n_mean.item()
                else:
                    nan_count += 1

        epoch_mean_loss = epoch_loss / total_batches
        epoch_accuracy = epoch_n_corrects / total_batches

        epoch_mean_normal_max = epoch_normal_max / (total_batches - nan_count)
        epoch_mean_normal_mean = epoch_normal_mean / (total_batches - nan_count)
        epoch_mean_abnormal_max = epoch_abnormal_max / total_batches
        epoch_mean_abnormal_mean = epoch_abnormal_mean / total_batches

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(f"==>> epoch {epoch+1} train_time: {train_time}\nloss: {round(epoch_mean_loss,4)}")
        print(f"accuracy: {epoch_accuracy:.2f}")
        print(f"==>> abnormal_max_mean: {epoch_mean_abnormal_max} abnormal_mean: {epoch_mean_abnormal_mean}")
        print(f"==>> normal_max_mean: {epoch_mean_normal_max} normal_mean: {epoch_mean_normal_mean}")

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

                total_n_corrects = 0

                total_max = 0
                total_mean = 0

                total_TP = 0
                total_FN = 0
                total_num_TRUE = 0
                total_FP = 0
                total_TN = 0
                total_num_FALSE = 0
                total_preds = []
                total_gts = []

                for step, (inputs, gts) in tqdm(
                    enumerate(concat_valid_loader), total=len(concat_valid_loader)
                ):
                    inputs = inputs.to(device)
                    # (val_batch_size, 11, 710)
                    # gts = gts.to(device)
                    # 이상영상이면 (val_batch_size, 176)
                    # 정상영상이면 (val_batch_size, 11)
                    if gts.size(1) == inputs.size(1):
                        gts11 = gts.view(-1, 1).to(device)
                        gts176 = torch.zeros(val_batch_size, gts.size(1) * 16)
                    else:
                        gts11 = (torch.max(gts.view(-1, inputs.size(1), 16), dim=2)[0]).view(-1, 1).to(device)
                        gts176 = gts

                    pred = model(inputs)
                    # pred는 (val_batch_size * 11, 1)

                    val_loss = criterion(pred, gts11)

                    pred_view = pred.view(val_batch_size, inputs.size(1))

                    pred_max = torch.mean(torch.max(pred_view, dim=-1)[0])

                    pred_mean = torch.mean(pred_view)

                    pred_correct = pred > thr
                    gts_correct = gts11 > thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    pred = (pred.squeeze()).detach().cpu().numpy()
                    pred_np = np.zeros(gts176.size(1))

                    step = np.array([i for i in range(gts176.size(1) + 1)])

                    for j in range(inputs.size(1)):
                        pred_np[step[j] * 16 : step[j + 1] * 16] = pred[j]

                    gts176 = gts176.squeeze().detach().numpy()

                    pred_positive = pred_np > thr
                    TP_and_FN = pred_positive[gts176 > 0.9]
                    FP_and_TN = pred_positive[gts176 < 0.1]

                    num_TP = np.sum(TP_and_FN)
                    num_FP = np.sum(FP_and_TN)

                    total_TP += num_TP
                    total_FN += len(TP_and_FN) - num_TP
                    total_FP += num_FP
                    total_TN += len(FP_and_TN) - num_FP

                    total_preds.append(pred_np)
                    total_gts.append(gts176)

                    total_num_TRUE += len(TP_and_FN)
                    total_num_FALSE += len(FP_and_TN)
                    total_n_corrects += corrects / inputs.size(1)

                    total_loss += val_loss.item()

                    total_max += pred_max.item()
                    total_mean += pred_mean.item()

                val_mean_loss = total_loss / len(concat_valid_loader)

                val_tpr = total_TP / total_num_TRUE
                val_fpr = total_FP / total_num_FALSE

                val_accuracy = total_n_corrects / len(concat_valid_loader)
                # for loop 한번에 abnormal 12, normal 12해서 24개 정답 확인

                val_mean_max = total_max / len(concat_valid_loader)
                val_mean_mean = total_mean / len(concat_valid_loader)

            if epoch == 0 or (epoch + 1) % (100 * val_interval) == 0:
                conf_mtx = np.array([[total_TP, total_FN], [total_FP, total_TN]])
                fig = plt.figure(figsize=(14, 7))
                ax = fig.add_subplot(121, aspect=1)
                sns.heatmap(
                    conf_mtx,
                    xticklabels=["Abnormal", "Normal"],
                    yticklabels=["Abnormal", "Normal"],
                    annot=True,
                    fmt="d",
                    ax=ax,
                )
                ax.set_title("Confusion matrix")
                ax.set_xlabel("Preds")
                ax.set_ylabel("GTs")

                ax2 = fig.add_subplot(122, aspect=1)
                total_preds_np = np.concatenate(total_preds)
                total_gts_np = np.concatenate(total_gts)

                fpr, tpr, cut = roc_curve(y_true=total_gts_np, y_score=total_preds_np)

                diff = tpr - fpr
                diff_idx = np.argmax(diff)
                best_thr = cut[diff_idx]
                val_bthr = best_thr if diff_idx != 0 else 1
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

                ax2.plot(fpr, tpr, linewidth=5, label=f"AUC = {val_auc}")
                ax2.plot([0, 1], [0, 1], linewidth=5)

                ax2.set_xlim([-0.01, 1])
                ax2.set_ylim([0, 1.01])
                ax2.legend(loc="lower right")
                ax2.set_title("ROC curve")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_xlabel("False Positive Rate")

                wandb.log({f"Confusion matrix and ROC": wandb.Image(fig, caption=f"{epoch+1} epoch")})
                # https://stackoverflow.com/questions/72134168/how-does-one-save-a-plot-in-wandb-with-wandb-log

                plt.clf()
                plt.close()

                # if epoch + 1 == max_epoch:
                print("Drawing graphs...")
                for i, (pred_g, gt_g) in tqdm(enumerate(zip(total_preds, total_gts)), total=len(total_preds)):
                    fig = plt.figure(figsize=(7, 7))
                    ax = fig.add_subplot(111)
                    ax.set_ylim([0, 1.01])
                    ax.set_title(f"Video {i} score graph")
                    ax.set_ylabel("Pred score")
                    ax.set_xlabel("Segment")
                    lins = np.linspace(0, len(pred_g) - 1, len(pred_g))

                    checker = np.concatenate((gt_g[1:], [0]))

                    gt_indices = np.where(gt_g != checker)[0]

                    for idx in range(len(gt_indices) // 2):

                        ax.axvspan(
                            gt_indices[2 * idx] + 0.5,
                            gt_indices[2 * idx + 1] + 0.5,
                            color="gray",
                            linestyle="--",
                            zorder=0,
                            alpha=0.3,
                        )

                    ax.plot(lins, pred_g)

                    if not os.path.exists(f"./graphs/{wandb_run_name}/{epoch}/"):
                        os.makedirs(f"./graphs/{wandb_run_name}/{epoch}/")

                    plt.savefig(f"./graphs/{wandb_run_name}/{epoch}/{i}.jpg", format="jpeg")

                    plt.clf()
                    plt.close()
            else:
                total_preds_np = np.concatenate(total_preds)
                total_gts_np = np.concatenate(total_gts)

                fpr, tpr, cut = roc_curve(y_true=total_gts_np, y_score=total_preds_np)

                diff = tpr - fpr
                diff_idx = np.argmax(diff)
                best_thr = cut[diff_idx]
                val_bthr = best_thr if diff_idx != 0 else 1
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

            if best_loss > val_mean_loss:
                print(f"Best performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_loss:.4f}")
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

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best.pth")
                torch.save(states, best_ckpt_fpath)
                best_loss = val_mean_loss
                # counter = 0
            # else:
            #     counter += 1

            if best_auc < val_auc:
                str_to_keep = f"Best auc performance at epoch: {epoch + 1}, {best_auc:.4f} -> {val_auc:.4f}"
                print(str_to_keep)
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

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best_auc.pth")
                torch.save(states, best_ckpt_fpath)
                best_auc = val_auc
                counter = 0
            else:
                counter += 1

        new_wandb_metric_dict = {
            "train_loss": epoch_mean_loss,
            "train_accuracy": epoch_accuracy,
            "valid_loss": val_mean_loss,
            "valid_fpr": val_fpr,
            "valid_tpr": val_tpr,
            "valid_bthr": val_bthr,
            "valid_auc": val_auc,
            "valid_ap": val_ap,
            "valid_accuracy": val_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_abnormal_max_mean": epoch_mean_abnormal_max,
            "train_abnormal_mean": epoch_mean_abnormal_mean,
            "train_normal_max_mean": epoch_mean_normal_max,
            "train_normal_mean": epoch_mean_normal_mean,
            "valid_max_mean": val_mean_max,
            "valid_mean": val_mean_mean,
        }

        wandb.log(new_wandb_metric_dict)

        scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print(f"==>> epoch {epoch+1} time: {epoch_time}\nvalid_loss: {round(val_mean_loss,4)}")
        print(f"valid_tpr: {val_tpr} valid_fpr: {val_fpr} valid_bthr: {val_bthr}")
        print(f"valid_auc: {val_auc:.4f} valid_ap: {val_ap:.4f} valid_accuracy: {val_accuracy:.2f}")
        print(f"==>> val_max_mean: {val_mean_max} val_mean: {val_mean_mean}")

        if counter > patience:
            print("Early Stopping...")
            break

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(str_to_keep)
    print(f"==>> total time: {total_time}")


def train3(
    normal_root_dir,
    abnormal_root_dir,
    json_dir,
    model_dir,
    model_name,
    model_size,
    device,
    num_workers,
    batch_size,
    # val_num_workers,
    # val_batch_size,
    learning_rate,
    max_epoch,
    val_interval,
    save_interval,
    thr,
    drop_rate,
    patience,
    resume_name,
    seed,
    # mp,
    use_extra,
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = batch_size

    val_batch_size = 1
    val_num_workers = 0

    # -- early stopping flag
    patience = patience
    counter = 0

    # 데이터셋
    dataset = NormalVMAE(
        model_size=model_size,
        root=normal_root_dir,
    )

    valid_data_size = len(dataset) // 10

    train_data_size = len(dataset) - valid_data_size

    train_dataset, valid_dataset = random_split(dataset, lengths=[train_data_size, valid_data_size])

    normal_train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )

    # normal_valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=val_batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     num_workers=val_num_workers,
    # )

    abnormal_train_dataset = AbnormalVMAE(
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=json_dir,
    )
    abnormal_valid_dataset = AbnormalVMAE(
        is_train=0,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=json_dir,
    )

    abnormal_train_loader = DataLoader(
        dataset=abnormal_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # abnormal_valid_loader = DataLoader(
    #     dataset=abnormal_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    # )

    concat_valid_dataset = ConcatDataset([valid_dataset, abnormal_valid_dataset])
    concat_valid_loader = DataLoader(
        dataset=concat_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> {model_size} data_load_time: {data_load_time}")

    # Initialize the model
    model = MILClassifier(drop_p=drop_rate)

    load_dict = None

    if resume_name:
        load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
        model.load_state_dict(load_dict["model_state_dict"])

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # 1e-6 => 0.0010000000474974513
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)

    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.BCELoss()
    MIL_criterion = MIL

    print(f"Start training..")

    wandb.init(
        project="VAD",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "무인매장",
            "n_epochs": max_epoch,
            "loss": "BCE+MIL",
            "notes": "VAD 실험",
        },
        name=wandb_run_name + "_" + train_start,
        mode=wandb_mode,
    )

    wandb.watch((model,))

    best_loss = np.inf
    best_auc = 0

    total_batches = len(abnormal_train_loader)

    for epoch in range(max_epoch):
        model.train()

        epoch_start = datetime.now()

        epoch_loss = 0
        epoch_n_corrects = 0
        epoch_MIL_loss = 0

        epoch_abnormal_max = 0
        epoch_abnormal_mean = 0
        epoch_normal_max = 0
        epoch_normal_mean = 0

        for step, abnormal_inputs in tqdm(
            enumerate(abnormal_train_loader),
            total=total_batches,
        ):
            if step % len(normal_train_loader) == 0:
                norm_train_iter = iter(normal_train_loader)
            # 중복 추출하더라도 정상, 이상 영상 1대1 대응 loop 끝까지 유지

            normal_inputs = next(norm_train_iter)

            abnormal_input, abnormal_gt = abnormal_inputs
            # (batch_size, 12, 710), (batch_size, 12)
            normal_input, normal_gt = normal_inputs
            # (batch_size, 12, 710), (batch_size, 12)

            inputs, gts = torch.cat((abnormal_input, normal_input), dim=1), torch.cat(
                (abnormal_gt, normal_gt), dim=1
            )
            # inputs는 (batch_size, 24, 710), gts는 (batch_size, 24)

            # batch_size = inputs.shape[0]

            # inputs = inputs.view(-1, inputs.size(-1)).to(device)
            # (batch_size * 24, 710)
            inputs = inputs.to(device)
            gts = gts.view(-1, 1).to(device)
            # (batch_size * 24, 1)

            optimizer.zero_grad()

            pred = model(inputs)
            # pred는 (batch_size * 24, 1)

            loss = criterion(pred, gts)
            MIL_loss = MIL_criterion(pred, batch_size, abnormal_input.size(1))
            sum_loss = loss + MIL_loss
            # sum_loss = MIL_loss
            sum_loss.backward()

            # loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_a = pred.view(batch_size, 2, abnormal_input.size(1))[:, 0, :]
                pred_n = pred.view(batch_size, 2, abnormal_input.size(1))[:, 1, :]

                pred_a_max = torch.mean(torch.max(pred_a, dim=-1)[0])
                pred_n_max = torch.mean(torch.max(pred_n, dim=-1)[0])

                pred_a_mean = torch.mean(pred_a)
                pred_n_mean = torch.mean(pred_n)

                pred_correct = pred > thr
                gts_correct = gts > thr

                pred_correct = pred_correct == gts_correct
                corrects = torch.sum(pred_correct).item()

                epoch_loss += loss.item()
                epoch_MIL_loss += MIL_loss.item()

                epoch_n_corrects += corrects / (abnormal_input.size(1) * 2)

                epoch_abnormal_max += pred_a_max.item()
                epoch_abnormal_mean += pred_a_mean.item()
                epoch_normal_max += pred_n_max.item()
                epoch_normal_mean += pred_n_mean.item()

        epoch_mean_loss = epoch_loss / total_batches
        epoch_mean_MIL_loss = epoch_MIL_loss / total_batches
        epoch_accuracy = epoch_n_corrects / (batch_size * (total_batches))

        epoch_mean_normal_max = epoch_normal_max / total_batches
        epoch_mean_normal_mean = epoch_normal_mean / total_batches
        epoch_mean_abnormal_max = epoch_abnormal_max / total_batches
        epoch_mean_abnormal_mean = epoch_abnormal_mean / total_batches

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(
            f"==>> epoch {epoch+1} train_time: {train_time}\nloss: {round(epoch_mean_loss,4)} MIL_loss: {round(epoch_mean_MIL_loss,4)}"
        )
        print(f"accuracy: {epoch_accuracy:.2f}")
        print(f"==>> abnormal_max_mean: {epoch_mean_abnormal_max} abnormal_mean: {epoch_mean_abnormal_mean}")
        print(f"==>> normal_max_mean: {epoch_mean_normal_max} normal_mean: {epoch_mean_normal_mean}")

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

                total_n_corrects = 0

                total_max = 0
                total_mean = 0

                total_TP = 0
                total_FN = 0
                total_num_TRUE = 0
                total_FP = 0
                total_TN = 0
                total_num_FALSE = 0
                total_preds = []
                total_gts = []

                for step, (inputs, gts) in tqdm(
                    enumerate(concat_valid_loader), total=len(concat_valid_loader)
                ):
                    inputs = inputs.to(device)
                    # (val_batch_size, 11, 710)
                    # gts = gts.to(device)
                    # 이상영상이면 (val_batch_size, 176)
                    # 정상영상이면 (val_batch_size, 11)
                    if gts.size(1) == inputs.size(1):
                        gts11 = gts.view(-1, 1).to(device)
                        gts176 = torch.zeros(val_batch_size, gts.size(1) * 16)
                    else:
                        gts11 = (torch.max(gts.view(-1, inputs.size(1), 16), dim=2)[0]).view(-1, 1).to(device)
                        gts176 = gts

                    pred = model(inputs)
                    # pred는 (val_batch_size * 11, 1)

                    val_loss = criterion(pred, gts11)

                    pred_view = pred.view(val_batch_size, inputs.size(1))

                    pred_max = torch.mean(torch.max(pred_view, dim=-1)[0])

                    pred_mean = torch.mean(pred_view)

                    pred_correct = pred > thr
                    gts_correct = gts11 > thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    pred = (pred.squeeze()).detach().cpu().numpy()
                    pred_np = np.zeros(gts176.size(1))

                    step = np.array([i for i in range(gts176.size(1) + 1)])

                    for j in range(inputs.size(1)):
                        pred_np[step[j] * 16 : step[j + 1] * 16] = pred[j]

                    gts176 = gts176.squeeze().detach().numpy()

                    pred_positive = pred_np > thr
                    TP_and_FN = pred_positive[gts176 > 0.9]
                    FP_and_TN = pred_positive[gts176 < 0.1]

                    num_TP = np.sum(TP_and_FN)
                    num_FP = np.sum(FP_and_TN)

                    total_TP += num_TP
                    total_FN += len(TP_and_FN) - num_TP
                    total_FP += num_FP
                    total_TN += len(FP_and_TN) - num_FP

                    total_preds.append(pred_np)
                    total_gts.append(gts176)

                    total_num_TRUE += len(TP_and_FN)
                    total_num_FALSE += len(FP_and_TN)
                    total_n_corrects += corrects / inputs.size(1)

                    total_loss += val_loss.item()

                    total_max += pred_max.item()
                    total_mean += pred_mean.item()

                val_mean_loss = total_loss / len(concat_valid_loader)

                val_tpr = total_TP / total_num_TRUE
                val_fpr = total_FP / total_num_FALSE

                val_accuracy = total_n_corrects / len(concat_valid_loader)
                # for loop 한번에 abnormal 12, normal 12해서 24개 정답 확인

                val_mean_max = total_max / len(concat_valid_loader)
                val_mean_mean = total_mean / len(concat_valid_loader)

            if epoch == 0 or (epoch + 1) % (100 * val_interval) == 0:
                conf_mtx = np.array([[total_TP, total_FN], [total_FP, total_TN]])
                fig = plt.figure(figsize=(14, 7))
                ax = fig.add_subplot(121, aspect=1)
                sns.heatmap(
                    conf_mtx,
                    xticklabels=["Abnormal", "Normal"],
                    yticklabels=["Abnormal", "Normal"],
                    annot=True,
                    fmt="d",
                    ax=ax,
                )
                ax.set_title("Confusion matrix")
                ax.set_xlabel("Preds")
                ax.set_ylabel("GTs")

                ax2 = fig.add_subplot(122, aspect=1)
                total_preds_np = np.concatenate(total_preds)
                total_gts_np = np.concatenate(total_gts)

                fpr, tpr, cut = roc_curve(y_true=total_gts_np, y_score=total_preds_np)

                diff = tpr - fpr
                diff_idx = np.argmax(diff)
                best_thr = cut[diff_idx]
                val_bthr = best_thr if diff_idx != 0 else 1
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

                ax2.plot(fpr, tpr, linewidth=5, label=f"AUC = {val_auc}")
                ax2.plot([0, 1], [0, 1], linewidth=5)

                ax2.set_xlim([-0.01, 1])
                ax2.set_ylim([0, 1.01])
                ax2.legend(loc="lower right")
                ax2.set_title("ROC curve")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_xlabel("False Positive Rate")

                wandb.log({f"Confusion matrix and ROC": wandb.Image(fig, caption=f"{epoch+1} epoch")})
                # https://stackoverflow.com/questions/72134168/how-does-one-save-a-plot-in-wandb-with-wandb-log

                plt.clf()
                plt.close()

                # if epoch + 1 == max_epoch:
                print("Drawing graphs...")
                for i, (pred_g, gt_g) in tqdm(enumerate(zip(total_preds, total_gts)), total=len(total_preds)):
                    fig = plt.figure(figsize=(7, 7))
                    ax = fig.add_subplot(111)
                    ax.set_ylim([0, 1.01])
                    ax.set_title(f"Video {i} score graph")
                    ax.set_ylabel("Pred score")
                    ax.set_xlabel("Segment")
                    lins = np.linspace(0, len(pred_g) - 1, len(pred_g))

                    checker = np.concatenate((gt_g[1:], [0]))

                    gt_indices = np.where(gt_g != checker)[0]

                    for idx in range(len(gt_indices) // 2):

                        ax.axvspan(
                            gt_indices[2 * idx] + 0.5,
                            gt_indices[2 * idx + 1] + 0.5,
                            color="gray",
                            linestyle="--",
                            zorder=0,
                            alpha=0.3,
                        )

                    ax.plot(lins, pred_g)

                    if not os.path.exists(f"./graphs/{wandb_run_name}/{epoch}/"):
                        os.makedirs(f"./graphs/{wandb_run_name}/{epoch}/")

                    plt.savefig(f"./graphs/{wandb_run_name}/{epoch}/{i}.jpg", format="jpeg")

                    plt.clf()
                    plt.close()
            else:
                total_preds_np = np.concatenate(total_preds)
                total_gts_np = np.concatenate(total_gts)

                fpr, tpr, cut = roc_curve(y_true=total_gts_np, y_score=total_preds_np)

                diff = tpr - fpr
                diff_idx = np.argmax(diff)
                best_thr = cut[diff_idx]
                val_bthr = best_thr if diff_idx != 0 else 1
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

            if best_loss > val_mean_loss:
                print(f"Best performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_loss:.4f}")
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

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best.pth")
                torch.save(states, best_ckpt_fpath)
                best_loss = val_mean_loss
                # counter = 0
            # else:
            #     counter += 1

            if best_auc < val_auc:
                str_to_keep = f"Best auc performance at epoch: {epoch + 1}, {best_auc:.4f} -> {val_auc:.4f}"
                print(str_to_keep)
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

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best_auc.pth")
                torch.save(states, best_ckpt_fpath)
                best_auc = val_auc
                counter = 0
            else:
                counter += 1

        new_wandb_metric_dict = {
            "train_loss": epoch_mean_loss,
            "train_accuracy": epoch_accuracy,
            "train_MIL_loss": epoch_mean_MIL_loss,
            "valid_loss": val_mean_loss,
            "valid_fpr": val_fpr,
            "valid_tpr": val_tpr,
            "valid_bthr": val_bthr,
            "valid_auc": val_auc,
            "valid_ap": val_ap,
            "valid_accuracy": val_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_abnormal_max_mean": epoch_mean_abnormal_max,
            "train_abnormal_mean": epoch_mean_abnormal_mean,
            "train_normal_max_mean": epoch_mean_normal_max,
            "train_normal_mean": epoch_mean_normal_mean,
            "valid_max_mean": val_mean_max,
            "valid_mean": val_mean_mean,
        }

        wandb.log(new_wandb_metric_dict)

        scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print(f"==>> epoch {epoch+1} time: {epoch_time}\nvalid_loss: {round(val_mean_loss,4)}")
        print(f"valid_tpr: {val_tpr} valid_fpr: {val_fpr} valid_bthr: {val_bthr}")
        print(f"valid_auc: {val_auc:.4f} valid_ap: {val_ap:.4f} valid_accuracy: {val_accuracy:.2f}")
        print(f"==>> val_max_mean: {val_mean_max} val_mean: {val_mean_mean}")

        if counter > patience:
            print("Early Stopping...")
            break

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(str_to_keep)
    print(f"==>> total time: {total_time}")


def main(args):
    if (args.wandb_run_name).startswith("MIL_t1"):
        print("train 1")
        train(**args.__dict__)
    elif (args.wandb_run_name).startswith("MIL_t2"):
        print("train 2")
        train2(**args.__dict__)
    else:
        print("train 3")
        train3(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()

    main(args)
