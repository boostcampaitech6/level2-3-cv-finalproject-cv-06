import cv2
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from timm.models import create_model

import albumentations as A

time_start = datetime.now()

time_start_str = time_start.strftime("%Y%m%d_%H%M%S")

segments_num = 8

backbone = create_model(
    "vit_small_patch16_224",
    img_size=224,
    pretrained=False,
    num_classes=710,
    all_frames=16 * segments_num,
    # tubelet_size=args.tubelet_size,
    # drop_rate=args.drop,
    # drop_path_rate=args.drop_path,
    # attn_drop_rate=args.attn_drop_rate,
    # head_drop_rate=args.head_drop_rate,
    # drop_block_rate=None,
    # use_mean_pooling=args.use_mean_pooling,
    # init_scale=args.init_scale,
    # with_cp=args.with_checkpoint,
)

load_dict = torch.load(
    "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/pths/vit_s_k710_dl_from_giant.pth"
)
# backbone pth 경로

backbone.load_state_dict(load_dict["module"])
model = nn.Sequential(backbone, nn.Linear(710, 1), nn.Sigmoid())
model.to("cuda")
model.eval()

tf = A.Resize(224, 224)

model_load_time = datetime.now()
load_time = model_load_time - time_start
load_time = str(load_time).split(".")[0]
print(f"==>> load_time: {load_time}")
# 시웅: 16*8 frame을 받는 모델 로드하는데 주피터 노트북 기준 11.7초
# => 모델을 이전에 미리 load 해놓으면 좋을 듯?

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
# 시웅: 제 컴퓨터 카메라는 frame 가로세로 변경이 안되서 일단 albumentations Resize를 사용했습니다.

threshold = 0.5

frames = []
count = 0


outputs_score = []
output_frames = []

while True:
    loop_start = datetime.now()

    ret, frame = cap.read()
    # frame.shape = (height, width, 3)
    if not ret:
        print("Cam Error")
        break

    frame = tf(image=frame)["image"]
    # frame.shape = (224, 224, 3)

    frame = np.expand_dims(frame, axis=0)
    # frame.shape = (1, 224, 224, 3)
    frames.append(frame)
    count += 1

    if count == 16 * segments_num:
        assert len(frames) == 16 * segments_num
        frames = np.concatenate(frames)
        # in_frames.shape = (16 * segments_num, 224, 224, 3)
        in_frames = frames.transpose(3, 0, 1, 2)
        # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
        in_frames = np.expand_dims(in_frames, axis=0)
        # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
        in_frames = torch.from_numpy(in_frames).float().to("cuda")
        # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])
        with torch.no_grad():
            output = model(in_frames)
            # output.shape == torch.Size([1, 1])

        output = torch.squeeze(output)
        output = output.item()
        outputs_score.append(output)
        if output >= threshold:
            output_frames.append(((len(outputs_score) - 1, frames.copy())))
        count = 0
        frames = []
        loop_end = datetime.now()
        loop_time = (loop_end - loop_start).total_seconds()
        print(f"Time to process {16*segments_num} frames: {loop_time * 1000:.0f} milliseconds")

time_end = datetime.now()
total_time = time_end - time_start
total_time = str(total_time).split(".")[0]
print(f"==>> total time: {total_time}")

cap.release()
cv2.destroyAllWindows()
