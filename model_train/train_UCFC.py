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

# from sklearn.preprocessing import MinMaxScaler

from argparse import ArgumentParser

import os
import os.path as osp

from tqdm import tqdm

import wandb

from shop_dataset import NewNormalVMAE, NewAbnormalVMAE
from classifier import WSAD, MILClassifier, MILandFE
from loss import MIL, MIL_top3, LossComputer


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--normal_root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_NORMAL_NPY",
            "../datapreprocess/npy/UCFCrime/normal",
        ),
    )
    # 학습 데이터 경로
    parser.add_argument(
        "--abnormal_root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_NPY",
            "../datapreprocess/npy/UCFCrime/abnormal",
        ),
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_ABNORMAL_LABEL",
            "../datapreprocess/npy/UCFCrime/test_anomalyv2.txt",
        ),
    )
    # abnormal 검증셋 npy, json파일 경로
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "../pths"))
    # pth 파일 저장 경로

    parser.add_argument("--model_name", type=str, default="BNWVAD")
    # import_module로 불러올 model name

    parser.add_argument("--len_feature", type=int, default=710)
    # npy파일 feature length
    parser.add_argument("--use_l2norm", action="store_true")
    # npy feature l2 normalization 여부
    parser.add_argument("--num_segments", type=int, default=200)
    # 영상 segment 개수

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
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00005)
    parser.add_argument("--max_epoch", type=int, default=1000)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--w_normal", type=float, default=1.0)
    parser.add_argument("--w_mpp", type=float, default=1.0)
    parser.add_argument("--gt_thr", type=float, default=0.25)
    parser.add_argument("--dist_thr", type=float, default=10)

    parser.add_argument("--ratio_sample", type=float, default=0.2)
    parser.add_argument("--ratio_batch", type=float, default=0.4)

    parser.add_argument("--ratios", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[1, 1, 1])

    parser.add_argument("--patience", type=int, default=100)

    # parser.add_argument("--mp", action="store_false")
    # https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    # mixed precision 사용할 지 여부

    # parser.add_argument("--use_extra", action="store_false")

    # parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    # wandb mode
    parser.add_argument("--wandb_run_name", type=str, default="BNWVAD")
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


def train_BNWVAD(
    normal_root_dir,
    abnormal_root_dir,
    label_dir,
    model_dir,
    model_name,
    model_size,
    device,
    num_workers,
    batch_size,
    # val_num_workers,
    # val_batch_size,
    learning_rate,
    weight_decay,
    max_epoch,
    val_interval,
    save_interval,
    w_normal,
    w_mpp,
    gt_thr,
    dist_thr,
    len_feature,
    use_l2norm,
    num_segments,
    ratio_sample,
    ratio_batch,
    ratios,
    kernel_sizes,
    patience,
    resume_name,
    seed,
    # mp,
    # use_extra,
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
    normal_train_dataset = NewNormalVMAE(
        is_train=1,
        model_size=model_size,
        root=normal_root_dir,
        num_segments=num_segments,
        l2_norm=use_l2norm,
    )
    # 800개
    normal_valid_dataset = NewNormalVMAE(
        is_train=0,
        model_size=model_size,
        root=normal_root_dir,
        num_segments=num_segments,
        l2_norm=use_l2norm,
    )
    # 149개

    normal_train_loader = DataLoader(
        dataset=normal_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # normal_valid_loader = DataLoader(
    #     dataset=normal_valid_dataset,
    #     batch_size=val_batch_size,
    #     shuffle=False,
    #     num_workers=val_num_workers,
    # )

    abnormal_train_dataset = NewAbnormalVMAE(
        is_train=1,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=label_dir,
        num_segments=num_segments,
        l2_norm=use_l2norm,
    )
    # 809개
    abnormal_valid_dataset = NewAbnormalVMAE(
        is_train=0,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=label_dir,
        num_segments=num_segments,
        gt_thr=gt_thr,
        l2_norm=use_l2norm,
    )
    # 140개

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

    concat_valid_dataset = ConcatDataset([normal_valid_dataset, abnormal_valid_dataset])
    concat_valid_loader = DataLoader(
        dataset=concat_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> {model_size} data_load_time: {data_load_time}")

    # Initialize the model
    model = WSAD(
        input_size=len_feature,
        ratio_sample=ratio_sample,
        ratio_batch=ratio_batch,
        ratios=ratios,
        kernel_sizes=kernel_sizes,
    )

    load_dict = None

    if resume_name:
        load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
        model.load_state_dict(load_dict["model_state_dict"])

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # 1e-6 => 0.0010000000474974513
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay
    )
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay
    # )
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)

    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.BCELoss()
    MPP_criterion = LossComputer(w_normal=w_normal, w_mpp=w_mpp)

    print(f"Start training..")

    wandb.init(
        project="VAD",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "무인매장",
            "n_epochs": max_epoch,
            "loss": "MPP",
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

        epoch_MPP_loss = 0
        epoch_norm_loss = 0
        epoch_MPP_and_norm_loss = 0

        epoch_abnormal_max = 0
        epoch_abnormal_mean = 0
        epoch_normal_max = 0
        epoch_normal_mean = 0

        for step, abnormal_input in tqdm(
            enumerate(abnormal_train_loader),
            total=total_batches,
        ):
            if step % len(normal_train_loader) == 0:
                norm_train_iter = iter(normal_train_loader)
            # 중복 추출하더라도 정상, 이상 영상 1대1 대응 loop 끝까지 유지

            normal_input = next(norm_train_iter)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            input = torch.cat((normal_input, abnormal_input), dim=1)
            # @@@@ BN-WVAD는 정상 영상 먼저 @@@@
            # inputs는 (batch_size, 2 * num_segments, 710)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            # batch_size = input.shape[0]

            input = input.to(device)

            optimizer.zero_grad()

            pred_result = model(input, flag="Train")
            # pred_result["pre_normal_scores"]: normal_scores[0 : b // 2],
            # pred_result["bn_results"]: bn_results,
            # pred_result["normal_scores"]: normal_scores,
            # pred_result["scores"]: distance_sum * normal_scores,

            pred = pred_result["scores"].view(-1, 1)
            # => pred는 (batch_size * 2 * num_segments, 1)

            MPP_and_norm_loss, loss_dict = MPP_criterion(pred_result)

            MPP_and_norm_loss.backward()

            # loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_n = pred.view(batch_size, 2, num_segments)[:, 0, :]
                pred_a = pred.view(batch_size, 2, num_segments)[:, 1, :]

                pred_n_max = torch.mean(torch.max(pred_n, dim=-1)[0])
                pred_a_max = torch.mean(torch.max(pred_a, dim=-1)[0])

                pred_n_mean = torch.mean(pred_n)
                pred_a_mean = torch.mean(pred_a)

                epoch_MPP_loss += loss_dict["mpp_loss"].item()
                epoch_norm_loss += loss_dict["normal_loss"].item()
                epoch_MPP_and_norm_loss += MPP_and_norm_loss.item()

                epoch_normal_max += pred_n_max.item()
                epoch_normal_mean += pred_n_mean.item()
                epoch_abnormal_max += pred_a_max.item()
                epoch_abnormal_mean += pred_a_mean.item()

        epoch_mean_MPP_loss = epoch_MPP_loss / total_batches
        epoch_mean_norm_loss = epoch_norm_loss / total_batches
        epoch_mean_MPP_and_norm_loss = epoch_MPP_and_norm_loss / total_batches

        epoch_mean_normal_max = epoch_normal_max / total_batches
        epoch_mean_normal_mean = epoch_normal_mean / total_batches
        epoch_mean_abnormal_max = epoch_abnormal_max / total_batches
        epoch_mean_abnormal_mean = epoch_abnormal_mean / total_batches

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(f"==>> epoch {epoch+1} train_time: {train_time}")
        print(
            f"MPP_loss: {round(epoch_mean_MPP_loss,4)} norm_loss: {round(epoch_mean_norm_loss,4)} MPP+norm_loss: {round(epoch_mean_MPP_and_norm_loss,4)}"
        )
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
                    # (val_batch_size, num_segments, 710)
                    gts = gts.view(-1, 1).to(device)
                    # (val_batch_size * num_segments, 1)

                    pred_result = model(inputs, flag="Eval_MPP")
                    # pred_result["normal_scores"]: normal_scores,
                    # pred_result["scores"]: distance_sum * normal_scores,
                    # breakpoint()
                    pred_acc = pred_result["normal_scores"].view(-1, 1)
                    pred = pred_result["scores"].view(-1, 1)
                    # pred는(batch_size * num_segments, 1)

                    val_loss = criterion(pred_acc, gts)

                    pred_view = pred.view(val_batch_size, num_segments)

                    pred_max = torch.mean(torch.max(pred_view, dim=-1)[0])

                    pred_mean = torch.mean(pred_view)

                    pred_correct = pred > dist_thr
                    gts_correct = gts  # > gt_thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    pred_np = (pred.squeeze()).detach().cpu().numpy()
                    gts_np = (gts.squeeze()).detach().cpu().numpy()
                    # pred_np, gts_np 둘다 (batch_size * num_segments)

                    pred_positive = pred_np > dist_thr
                    TP_and_FN = pred_positive[gts_np > 0.9]
                    FP_and_TN = pred_positive[gts_np < 0.1]

                    num_TP = np.sum(TP_and_FN)
                    num_FP = np.sum(FP_and_TN)

                    # if epoch == 0 or (epoch + 1) % (100 * val_interval) == 0:
                    total_TP += num_TP
                    total_FN += len(TP_and_FN) - num_TP
                    total_FP += num_FP
                    total_TN += len(FP_and_TN) - num_FP

                    total_preds.append(pred_np.copy())
                    total_gts.append(gts_np.copy())

                    total_num_TRUE += len(TP_and_FN)
                    total_num_FALSE += len(FP_and_TN)
                    total_n_corrects += corrects / num_segments

                    total_loss += val_loss.item()

                    total_max += pred_max.item()
                    total_mean += pred_mean.item()

                val_mean_loss = total_loss / len(concat_valid_loader)

                val_tpr = total_TP / total_num_TRUE
                val_fpr = total_FP / total_num_FALSE

                val_accuracy = total_n_corrects / len(concat_valid_loader)

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
                val_bthr = best_thr if diff_idx != 0 else (cut[1] * 10)
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
                val_bthr = best_thr if diff_idx != 0 else (cut[1] * 10)
                # 무한대값 처리

                precision, recall, cut2 = precision_recall_curve(total_gts_np, total_preds_np)

                val_auc = sklearn.metrics.auc(fpr, tpr)
                val_ap = sklearn.metrics.auc(recall, precision)

            if best_loss > val_mean_loss:
                print(f"Best loss performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_loss:.4f}")
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
            "train_MPP_loss": epoch_mean_MPP_loss,
            "train_norm_loss": epoch_mean_norm_loss,
            "train_MPP+norm_loss": epoch_mean_MPP_and_norm_loss,
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
        # print(
        #     f"valid_n_MPP_loss: {round(val_n_mean_MPP_loss,4)} valid_n_norm_loss: {round(val_n_mean_norm_loss,4)} valid_n_MPP+norm_loss: {round(val_n_mean_MPP_and_norm_loss,4)}"
        # )
        print(f"valid_fpr: {val_fpr} valid_tpr: {val_tpr} valid_bthr: {val_bthr}")
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


def train_MIL(
    normal_root_dir,
    abnormal_root_dir,
    label_dir,
    model_dir,
    model_name,
    model_size,
    device,
    num_workers,
    batch_size,
    # val_num_workers,
    # val_batch_size,
    learning_rate,
    weight_decay,
    max_epoch,
    val_interval,
    save_interval,
    w_normal,
    w_mpp,
    gt_thr,
    dist_thr,
    len_feature,
    use_l2norm,
    num_segments,
    ratio_sample,
    ratio_batch,
    ratios,
    kernel_sizes,
    patience,
    resume_name,
    seed,
    # mp,
    # use_extra,
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
    normal_train_dataset = NewNormalVMAE(
        is_train=1,
        model_size=model_size,
        root=normal_root_dir,
        num_segments=num_segments,
        l2_norm=use_l2norm,
    )
    # 800개
    normal_valid_dataset = NewNormalVMAE(
        is_train=0,
        model_size=model_size,
        root=normal_root_dir,
        num_segments=num_segments,
        l2_norm=use_l2norm,
    )
    # 149개

    normal_train_loader = DataLoader(
        dataset=normal_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # normal_valid_loader = DataLoader(
    #     dataset=normal_valid_dataset,
    #     batch_size=val_batch_size,
    #     shuffle=False,
    #     num_workers=val_num_workers,
    # )

    abnormal_train_dataset = NewAbnormalVMAE(
        is_train=1,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=label_dir,
        num_segments=num_segments,
        l2_norm=use_l2norm,
    )
    # 809개
    abnormal_valid_dataset = NewAbnormalVMAE(
        is_train=0,
        model_size=model_size,
        root=abnormal_root_dir,
        label_root=label_dir,
        num_segments=num_segments,
        gt_thr=gt_thr,
        l2_norm=use_l2norm,
    )
    # 140개

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

    concat_valid_dataset = ConcatDataset([normal_valid_dataset, abnormal_valid_dataset])
    concat_valid_loader = DataLoader(
        dataset=concat_valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> {model_size} data_load_time: {data_load_time}")

    # Initialize the model
    model = MILClassifier(drop_p=0.6)
    # model = MILandFE(drop_p=0.3)

    load_dict = None

    if resume_name:
        load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
        model.load_state_dict(load_dict["model_state_dict"])

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # 1e-6 => 0.0010000000474974513
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay
    # )
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0010000000474974513)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)

    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.BCELoss()
    MIL_criterion = MIL
    # MIL_criterion = MIL_top3

    print(f"Start training..")

    wandb.init(
        project="VAD",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "무인매장",
            "n_epochs": max_epoch,
            "loss": "MIL",
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

        epoch_MIL_loss = 0

        epoch_abnormal_max = 0
        epoch_abnormal_mean = 0
        epoch_normal_max = 0
        epoch_normal_mean = 0

        for step, abnormal_input in tqdm(
            enumerate(abnormal_train_loader),
            total=total_batches,
        ):
            if step % len(normal_train_loader) == 0:
                norm_train_iter = iter(normal_train_loader)
            # 중복 추출하더라도 정상, 이상 영상 1대1 대응 loop 끝까지 유지

            normal_input = next(norm_train_iter)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            input = torch.cat((abnormal_input, normal_input), dim=1)
            # @@@@ MIL은 이상 영상 먼저 @@@@
            # inputs는 (batch_size, 2 * num_segments, 710)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            # batch_size = input.shape[0]

            input = input.to(device)

            optimizer.zero_grad()

            pred = model(input)
            # pred는 (batch_size * 2 * num_segments, 1)

            MIL_loss = MIL_criterion(pred, batch_size, num_segments)

            MIL_loss.backward()

            # loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_a = pred.view(batch_size, 2, num_segments)[:, 0, :]
                pred_n = pred.view(batch_size, 2, num_segments)[:, 1, :]

                pred_a_max = torch.mean(torch.max(pred_a, dim=-1)[0])
                pred_n_max = torch.mean(torch.max(pred_n, dim=-1)[0])

                pred_a_mean = torch.mean(pred_a)
                pred_n_mean = torch.mean(pred_n)

                epoch_MIL_loss += MIL_loss.item()

                epoch_normal_max += pred_n_max.item()
                epoch_normal_mean += pred_n_mean.item()
                epoch_abnormal_max += pred_a_max.item()
                epoch_abnormal_mean += pred_a_mean.item()

        epoch_mean_MIL_loss = epoch_MIL_loss / total_batches

        epoch_mean_normal_max = epoch_normal_max / total_batches
        epoch_mean_normal_mean = epoch_normal_mean / total_batches
        epoch_mean_abnormal_max = epoch_abnormal_max / total_batches
        epoch_mean_abnormal_mean = epoch_abnormal_mean / total_batches

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(f"==>> epoch {epoch+1} train_time: {train_time}")
        print(f"MIL_loss: {round(epoch_mean_MIL_loss,4)}")
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
                    # (val_batch_size, num_segments, 710)
                    gts = gts.view(-1, 1).to(device)
                    # (val_batch_size * num_segments, 1)

                    pred = model(inputs)
                    # pred는 (val_batch_size * num_segments, 1)

                    val_loss = criterion(pred, gts)

                    pred_view = pred.view(val_batch_size, num_segments)

                    pred_max = torch.mean(torch.max(pred_view, dim=-1)[0])

                    pred_mean = torch.mean(pred_view)

                    pred_correct = pred > dist_thr
                    gts_correct = gts  # > gt_thr

                    pred_correct = pred_correct == gts_correct
                    corrects = torch.sum(pred_correct).item()

                    pred_np = (pred.squeeze()).detach().cpu().numpy()
                    gts_np = (gts.squeeze()).detach().cpu().numpy()
                    # pred_np, gts_np 둘다 (batch_size * num_segments)

                    pred_positive = pred_np > dist_thr
                    TP_and_FN = pred_positive[gts_np > 0.9]
                    FP_and_TN = pred_positive[gts_np < 0.1]

                    num_TP = np.sum(TP_and_FN)
                    num_FP = np.sum(FP_and_TN)

                    # if epoch == 0 or (epoch + 1) % (100 * val_interval) == 0:
                    total_TP += num_TP
                    total_FN += len(TP_and_FN) - num_TP
                    total_FP += num_FP
                    total_TN += len(FP_and_TN) - num_FP

                    total_preds.append(pred_np.copy())
                    total_gts.append(gts_np.copy())

                    total_num_TRUE += len(TP_and_FN)
                    total_num_FALSE += len(FP_and_TN)
                    total_n_corrects += corrects / num_segments

                    total_loss += val_loss.item()

                    total_max += pred_max.item()
                    total_mean += pred_mean.item()

                val_mean_loss = total_loss / len(concat_valid_loader)

                val_tpr = total_TP / total_num_TRUE
                val_fpr = total_FP / total_num_FALSE

                val_accuracy = total_n_corrects / len(concat_valid_loader)

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
                print(f"Best loss performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_loss:.4f}")
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
        # print(
        #     f"valid_n_MPP_loss: {round(val_n_mean_MPP_loss,4)} valid_n_norm_loss: {round(val_n_mean_norm_loss,4)} valid_n_MPP+norm_loss: {round(val_n_mean_MPP_and_norm_loss,4)}"
        # )
        print(f"valid_fpr: {val_fpr} valid_tpr: {val_tpr} valid_bthr: {val_bthr}")
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
    if (args.model_name) == "BNWVAD":
        train_BNWVAD(**args.__dict__)
    elif (args.model_name) == "MIL":
        train_MIL(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()

    main(args)
