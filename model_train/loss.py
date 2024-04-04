import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

import pdb


def MIL(y_pred, batch_size, feature_length, is_transformer=0):
    loss = torch.tensor(0.0).cuda()
    loss_intra = torch.tensor(0.0).cuda()
    sparsity = torch.tensor(0.0).cuda()
    smooth = torch.tensor(0.0).cuda()
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
        # (30*24, 1)을 (30, 24)로 다시 변경
        # dim=1 24 = 이상12 + 정상12
    else:
        y_pred = torch.sigmoid(y_pred)

    # print(f"==>> y_pred.shape: {y_pred.shape}")

    for i in range(batch_size):
        # anomaly_index = torch.randperm(12).cuda()
        # print(f"==>> anomaly_index: {anomaly_index}")
        # normal_index = torch.randperm(12).cuda()
        # print(f"==>> normal_index: {normal_index}")

        # print(f"==>> y_pred[i, :12].shape: {y_pred[i, :12].shape}")
        # print(f"==>> y_pred[i, 12:].shape: {y_pred[i, 12:].shape}")

        y_anomaly = y_pred[i, :feature_length]
        # y_anomaly = y_pred[i, :12][anomaly_index]
        # print(f"==>> y_anomaly.shape: {y_anomaly.shape}")
        # MIL 논문의 segment 개수 32와 다르게 무인매장 데이터셋 feature는 12 segment
        y_normal = y_pred[i, feature_length:]
        # y_normal = y_pred[i, 12:][normal_index]
        # print(f"==>> y_normal.shape: {y_normal.shape}")

        y_anomaly_max = torch.max(y_anomaly)  # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)  # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.0 - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, : feature_length - 1] - y_pred[i, 1:feature_length]) ** 2) * 0.00008
    loss = (loss + sparsity + smooth) / batch_size

    return loss


def MIL_top3(y_pred, batch_size, feature_length, is_transformer=0):
    # https://arxiv.org/abs/2002.01132
    loss = torch.tensor(0.0).cuda()
    sparsity = torch.tensor(0.0).cuda()
    smooth = torch.tensor(0.0).cuda()
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
        # (30*2*feature_length, 1)을 (30, 2*feature_length)로 다시 변경
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):

        y_anomaly = y_pred[i, :feature_length]

        y_normal = y_pred[i, feature_length:]

        y_anomaly_top3 = torch.topk(y_anomaly, 3)[0]  # anomaly top3
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)  # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.0 - y_anomaly_top3[0] + y_normal_max)
        loss += F.relu(1.0 - y_anomaly_top3[1] + y_normal_max)
        loss += F.relu(1.0 - y_anomaly_top3[2] + y_normal_max)
        loss += F.relu(1.0 - y_anomaly_top3[0] + y_anomaly_min)

        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, : feature_length - 1] - y_pred[i, 1:feature_length]) ** 2) * 0.00008
    loss = (loss + sparsity + smooth) / batch_size

    return loss


class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, normal_scores):
        """
        normal_scores: [bs, pre_k]
        """
        loss_normal = torch.norm(normal_scores, dim=1, p=2)
        # normal_scores는 정상영상의 snippet score만 있는 상태 => (n_batch_size, t snippets)

        return loss_normal.mean()


class MPPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_triplet = [5, 20]

    def forward(self, anchors, variances, select_normals, select_abnormals):
        losses_triplet = []

        def mahalanobis_distance(mu, x, var):
            return torch.sqrt(torch.sum((x - mu) ** 2 / var, dim=-1))

        for anchor, var, pos, neg, wt in zip(
            anchors, variances, select_normals, select_abnormals, self.w_triplet
        ):
            triplet_loss = nn.TripletMarginWithDistanceLoss(
                margin=1, distance_function=partial(mahalanobis_distance, var=var)
            )

            B, C, k = pos.shape
            pos = pos.permute(0, 2, 1).reshape(B * k, -1)
            neg = neg.permute(0, 2, 1).reshape(B * k, -1)
            loss_triplet = triplet_loss(anchor[None, ...].repeat(B * k, 1), pos, neg)
            # pos, neg, anchor[None, ...].repeat(B * k, 1) 모두 (B * k, -1) 형태 동일
            losses_triplet.append(loss_triplet * wt)

        return sum(losses_triplet)


class LossComputer(nn.Module):
    def __init__(self, w_normal=1.0, w_mpp=1.0):
        super().__init__()
        self.w_normal = w_normal
        self.w_mpp = w_mpp
        self.mppLoss = MPPLoss()
        self.normalLoss = NormalLoss()

    def forward(self, result):
        loss = {}
        # breakpoint()
        pre_normal_scores = result["pre_normal_scores"]
        # (n_batch_size, t snippets) 형태
        normal_loss = self.normalLoss(pre_normal_scores)
        # normal_loss 계산에는 정상영상의 정상 snippet들 score만 사용
        # 논문 3.4 확인
        loss["normal_loss"] = normal_loss

        anchors = result["bn_results"]["anchors"]
        variances = result["bn_results"]["variances"]
        select_normals = result["bn_results"]["select_normals"]
        select_abnormals = result["bn_results"]["select_abnormals"]

        mpp_loss = self.mppLoss(anchors, variances, select_normals, select_abnormals)
        loss["mpp_loss"] = mpp_loss

        loss["total_loss"] = self.w_normal * normal_loss + self.w_mpp * mpp_loss

        return loss["total_loss"], loss
