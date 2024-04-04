# from collections import defaultdict
import argparse
import os
from copy import deepcopy
from datetime import datetime

# import torch.nn as nn
# import torch.nn.functional as F
import albumentations as A
import cv2
import matplotlib.pyplot as plt

# from datetime import datetime
import models
import numpy as np
import torch
from timm.models import create_model
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


parser = argparse.ArgumentParser(description="Feature Extraction")

parser.add_argument(
    "--root",
    type=str,
    help="root folder path",
    default="/data/ephemeral/home/datasets/UCFCrime/normal/",
)

args = parser.parse_args()


root = "/data/ephemeral/home/datasets/UCFCrime/normal/"

npy_root = "./npy/"

if not os.path.exists(npy_root):
    os.makedirs(npy_root)


folder_list = os.listdir(root)
folder_list.sort()
print(f"==>> folder_list: {folder_list}")

segments_num = 1
# 모델에 들어갈 frame수는 16 * segments_num

model = create_model(
    "vit_small_patch16_224",
    img_size=224,
    pretrained=False,
    num_classes=710,
    all_frames=16 * segments_num,
)

load_dict = torch.load(
    "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/vit_s_k710_dl_from_giant.pth"
)

model.load_state_dict(load_dict["module"])

model.to("cuda")
model.eval()

tf = A.Resize(224, 224)

for folder_name in folder_list:

    time_start = datetime.now()

    print(f"{folder_name} feature extracting starts")

    if not os.path.exists(npy_root + folder_name):
        os.makedirs(npy_root + folder_name)

    folder_path = root + folder_name + "/"

    file_list = os.listdir(root + folder_name)
    file_list.sort()
    print(f"==>> file_list: {file_list}")

    batch_size = 16
    # Loop through the video frames
    for file_name in tqdm(file_list, total=len(file_list)):
        path = folder_path + file_name

        cap = cv2.VideoCapture(path)

        # 710차원 feature array 저장할 list
        np_list = []

        # 16 * segments_num 프레임씩 저장할 list
        frames = []
        frame_count = 0

        # input tensor 저장할 list
        input_list = []
        input_count = 0

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            # frame.shape = (height, width, 3)

            frame_count += 1  # Increment frame count

            if success:
                frame = tf(image=frame)["image"]
                # frame.shape = (224, 224, 3)

                frame = np.expand_dims(frame, axis=0)
                # frame.shape = (1, 224, 224, 3)
                frames.append(frame.copy())

                if frame_count == 16 * segments_num:
                    assert len(frames) == 16 * segments_num
                    frames = np.concatenate(frames)
                    # in_frames.shape = (16 * segments_num, 224, 224, 3)
                    in_frames = frames.transpose(3, 0, 1, 2)
                    # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                    in_frames = np.expand_dims(in_frames, axis=0)
                    # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                    in_frames = torch.from_numpy(in_frames).float()
                    # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                    input_list.append(in_frames.detach().clone())

                    frame_count = 0
                    frames = []

                    input_count += 1

                    if input_count == batch_size:
                        # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                        input_batch = torch.cat(input_list, dim=0).to("cuda")

                        with torch.no_grad():
                            output = model(input_batch)
                            # output.shape == torch.Size([batch_size, 710])

                        np_list.append(output.cpu().numpy())

                        input_count = 0
                        input_list = []
            else:
                # 남은 프레임, input_list가 지정 개수에서 모자를 때 예외 처리
                if frame_count != 0 and len(frames) != 0:
                    # @@ success가 false 일때도 frame_count는 +1이 된다
                    # @@ => frames = []로 초기화 된 바로 다음 frame에 success가 false가 되면
                    # @@ => frame_count == 1 이지만 len(frames) == 0
                    len_frames_left = 16 * segments_num - len(frames)
                    # len_input_list_left = batch_size - len(input_list)

                    # assert len(frames) != 0

                    for i in range(len_frames_left):
                        try:
                            frames.append(frames[-1].copy())
                        except IndexError:
                            print(f"==>> len(frames): {len(frames)}")
                            print(f"==>> len_frames_left: {len_frames_left}")

                    assert len(frames) == 16 * segments_num

                    frames = np.concatenate(frames)
                    # in_frames.shape = (16 * segments_num, 224, 224, 3)
                    in_frames = frames.transpose(3, 0, 1, 2)
                    # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                    in_frames = np.expand_dims(in_frames, axis=0)
                    # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                    in_frames = torch.from_numpy(in_frames).float()
                    # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                    input_list.append(in_frames.detach().clone())

                    # assert len(input_list) == batch_size

                    # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                    input_batch = torch.cat(input_list, dim=0).to("cuda")

                    with torch.no_grad():
                        output = model(input_batch)
                        # output.shape == torch.Size([len(input_list), 710])

                    np_list.append(output.cpu().numpy())

                    frame_count = 0
                    frames = []
                    input_count = 0
                    input_list = []

                # Break the loop if the end of the video is reached
                break
        try:
            file_outputs = np.concatenate(np_list)
            # print(f"==>> file_outputs.shape: {file_outputs.shape}")
            np.save((npy_root + folder_name + "/" + file_name), file_outputs)
        except ValueError:
            print(f"{file_name} ValueError: need at least one array to concatenate")

        cap.release()

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]

    print(f"{folder_name} feature extracting ended. Elapsed time: {total_time}")

# segments_num = 1
# 모델에 들어갈 frame수는 16 * segments_num

model = create_model(
    "vit_base_patch16_224",
    img_size=224,
    pretrained=False,
    num_classes=710,
    all_frames=16 * segments_num,
)

load_dict = torch.load(
    "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/vit_b_k710_dl_from_giant.pth"
)
# backbone pth 경로

model.load_state_dict(load_dict["module"])

model.to("cuda")
model.eval()

tf = A.Resize(224, 224)

for folder_name in folder_list:
    time_start = datetime.now()

    print(f"{folder_name} feature extracting starts")

    if not os.path.exists(npy_root + folder_name + "_base"):
        os.makedirs(npy_root + folder_name + "_base")

    folder_path = root + folder_name + "/"

    file_list = os.listdir(root + folder_name)
    file_list.sort()
    print(f"==>> file_list: {file_list}")

    batch_size = 16
    # Loop through the video frames
    for file_name in tqdm(file_list, total=len(file_list)):
        path = folder_path + file_name

        cap = cv2.VideoCapture(path)

        # 710차원 feature array 저장할 list
        np_list = []

        # 16 * segments_num 프레임씩 저장할 list
        frames = []
        frame_count = 0

        # input tensor 저장할 list
        input_list = []
        input_count = 0

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            # frame.shape = (height, width, 3)

            frame_count += 1  # Increment frame count

            if success:
                frame = tf(image=frame)["image"]
                # frame.shape = (224, 224, 3)

                frame = np.expand_dims(frame, axis=0)
                # frame.shape = (1, 224, 224, 3)
                frames.append(frame.copy())

                if frame_count == 16 * segments_num:
                    assert len(frames) == 16 * segments_num
                    frames = np.concatenate(frames)
                    # in_frames.shape = (16 * segments_num, 224, 224, 3)
                    in_frames = frames.transpose(3, 0, 1, 2)
                    # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                    in_frames = np.expand_dims(in_frames, axis=0)
                    # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                    in_frames = torch.from_numpy(in_frames).float()
                    # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                    input_list.append(in_frames.detach().clone())

                    frame_count = 0
                    frames = []

                    input_count += 1

                    if input_count == batch_size:
                        # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                        input_batch = torch.cat(input_list, dim=0).to("cuda")

                        with torch.no_grad():
                            output = model(input_batch)
                            # output.shape == torch.Size([batch_size, 710])

                        np_list.append(output.cpu().numpy())

                        input_count = 0
                        input_list = []
            else:
                # 남은 프레임, input_list가 지정 개수에서 모자를 때 예외 처리
                if frame_count != 0 and len(frames) != 0:
                    # @@ success가 false 일때도 frame_count는 +1이 된다
                    # @@ => frames = []로 초기화 된 바로 다음 frame에 success가 false가 되면
                    # @@ => frame_count == 1 이지만 len(frames) == 0
                    len_frames_left = 16 * segments_num - len(frames)
                    # len_input_list_left = batch_size - len(input_list)
                    for i in range(len_frames_left):
                        frames.append(frames[-1].copy())

                    assert len(frames) == 16 * segments_num

                    frames = np.concatenate(frames)
                    # in_frames.shape = (16 * segments_num, 224, 224, 3)
                    in_frames = frames.transpose(3, 0, 1, 2)
                    # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                    in_frames = np.expand_dims(in_frames, axis=0)
                    # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                    in_frames = torch.from_numpy(in_frames).float()
                    # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                    input_list.append(in_frames.detach().clone())

                    # assert len(input_list) == batch_size

                    # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                    input_batch = torch.cat(input_list, dim=0).to("cuda")

                    with torch.no_grad():
                        output = model(input_batch)
                        # output.shape == torch.Size([len(input_list), 710])

                    np_list.append(output.cpu().numpy())

                    frame_count = 0
                    frames = []
                    input_count = 0
                    input_list = []

                # Break the loop if the end of the video is reached
                break

        try:
            file_outputs = np.concatenate(np_list)
            # print(f"==>> file_outputs.shape: {file_outputs.shape}")
            np.save((npy_root + folder_name + "_base/" + file_name), file_outputs)
        except ValueError:
            print(f"{file_name} ValueError: need at least one array to concatenate")

        cap.release()

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]

    print(f"{folder_name} feature extracting ended. Elapsed time: {total_time}")
