from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
from sklearn.preprocessing import MinMaxScaler

from fastapi import HTTPException
from starlette import status
from database import crud
from database import schemas

import os
import uuid
import json
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from io import BytesIO

import torch
import torch.nn as nn

import albumentations as A

import sys

sys.path.append("/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models")

import vmae

# @@ timm은 0.4.12 버전 사용 필수
from timm.models import create_model
from copy import deepcopy

class RT_AnomalyDetector:
    def __init__(self, info, s3_client, settings, db, websocket):
        self.info = info
        self.s3 = s3_client
        self.settings = settings
        self.frame_url_base = f"frame/{info['user_id']}/{info['upload_id']}/"
        self.db = db
        self.websocket = websocket

    async def upload_frame_db(self, db, temp_for_db, frame_url):

        temp_json_path = "./temp.json"

        with open(temp_json_path, "w") as f:
            json.dump(temp_for_db, f)

        with open(temp_json_path, "r") as f:
            box_kp_json = json.load(f)

        _frame_create = schemas.FrameCreate(
            frame_url=frame_url,
            time_stamp=temp_for_db["timestamp"],
            box_kp_json=box_kp_json,
            score=temp_for_db["score"],
            video_id=self.info["video_id"],
        )

        crud.create_frame(db=db, frame=_frame_create)

        os.remove(temp_json_path)

    async def upload_frame_s3(self, s3, frame):
        s3_upload_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Frame 을 s3 저장소 업로드에 실패했습니다."
        )
        frame_name = uuid.uuid1()
        frame_url = self.frame_url_base + f"{frame_name}" + ".png"
        # print(frame_url)

        try:
            s3.upload_fileobj(
                BytesIO(cv2.imencode(".png", frame)[1].tobytes()),
                self.settings.BUCKET,
                frame_url,
                ExtraArgs={"ContentType": "image/png"},
            )
        except Exception as e:
            # print(e)
            raise s3_upload_exception

        return frame_url

    def upload_score_graph_s3(self):
        plt.plot(self.scores, color="red")
        plt.title("Anomaly Scores Over Time")
        plt.xlabel(" ")
        plt.ylabel(" ")

        plt.xticks([])
        plt.yticks([])

        save_path = "./model_scores_plot.png"
        plt.savefig(save_path)

        with open(save_path, "rb") as image_file:
            graph = image_file.read()

        s3_upload_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="score Graph 를 s3 저장소 업로드에 실패했습니다."
        )
        score_graph_name = "score_graph.png"
        score_graph_url = self.frame_url_base + score_graph_name

        try:
            self.s3.upload_fileobj(
                BytesIO(graph), self.settings.BUCKET, score_graph_url, ExtraArgs={"ContentType": "image/png"}
            )
        except:
            raise s3_upload_exception

        os.remove(save_path)

    def ready(self):
        # YOLO
        self.tracker_model = YOLO("yolov8n-pose.pt")

        # VMAE v2
        self.backbone = model = create_model(
            "vit_base_patch16_224",
            img_size=224,
            pretrained=False,
            num_classes=710,
            all_frames=16,
        )

        load_dict = torch.load(
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pts/vit_b_k710_dl_from_giant.pth"
        )

        self.backbone.load_state_dict(load_dict["module"])

        self.tf = A.Resize(224, 224)

        # Define sequence_length, prediction_time, and n_features
        # sequence_length = 20
        # prediction_time = 1
        # n_features = 38

        # classifier
        checkpoint = torch.load(
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pts/MIL_20240318_180129_best_auc.pth"
        )
        self.classifier = vmae.MILClassifier(input_dim=710, drop_p=0.3)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker_model.to(device)
        self.backbone.to(device)
        self.backbone.eval()
        self.classifier.to(device)
        self.classifier.eval()

        # Store the track history
        self.track_history = defaultdict(lambda: [])

        # Initialize a dictionary to store separate buffers for each ID
        self.id_buffers = defaultdict(lambda: [])

        # Loop through the video frames
        self.frame_count = 0
        # self.net_mse = 0
        # self.avg_mse = 0

        # score graph 를 위한 score list
        self.scores = []

        # vmae에 입력할 frame들을 저장할 list
        self.v_frames = []

        # 영상 frame 임시 저장 list
        self.frame_list = []
        # yolo results 임시 저장 list
        self.results_list = []
        # temp_for_db 임시 저장 list
        self.tfdb_list = []

        # timestamp 저장
        self.prv_timestamp = 0
        self.fps3_delta = timedelta(seconds=1 / 3)

    async def run(self, frame, timestamp):

        # Define the standard frame size
        standard_width = 640
        standard_height = 480

        # Define sequence_length, prediction_time, and n_features
        # sequence_length = 20
        # prediction_time = 1
        # n_features = 38

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define a function to calculate MSE between two sequences
        def calculate_mse(seq1, seq2):
            return np.mean(np.power(seq1 - seq2, 2))

        # anomaly threshold (default 0.02)
        # threshold = self.info["thr"]
        threshold = 0.1

        self.frame_count += 1  # Increment frame count

        frame = cv2.resize(frame, (standard_width, standard_height))

        temp_for_db = {"timestamp": None, "bbox": {}, "keypoints": {}, "score": None}

        # track id (사람) 별로 mse 점수가 나오기 때문에 한 frame 에 여러 mse 점수가 나옴. 이를 frame 별 점수로 구하기 위해서 변수 설정
        # mse_unit = 0

        frame_checker = False

        # 이전에 frame을 저장한 시점에서 0.3333.. 초 이상 경과했는지 확인
        if self.prv_timestamp == 0:
            frame_checker = True
            self.prv_timestamp = timestamp
        else:
            time_delta = timestamp - self.prv_timestamp
            if time_delta > self.fps3_delta:
                frame_checker = True
                self.prv_timestamp = timestamp

        timestamp = timestamp.strftime("%H:%M:%S")
        temp_for_db["timestamp"] = timestamp

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.tracker_model.track(frame, persist=True)

        # frame_checker = True
        # 1초에 3 frame만 저장해서 vmae+MIL에 사용
        if frame_checker:
            self.frame_list.append(frame.copy())
            self.results_list.append(deepcopy(results))
            self.tfdb_list.append(deepcopy(temp_for_db))

            v_frame = self.tf(image=frame)["image"]
            # (224, 224, 3)
            v_frame = np.expand_dims(v_frame, axis=0)
            # (1, 224, 224, 3)
            self.v_frames.append(v_frame.copy())

            # 16 frame이 모이면 vmae+MIL 계산
            if len(self.v_frames) == 16:
                in_frames = np.concatenate(self.v_frames)
                # (16, 224, 224, 3)
                in_frames = in_frames.transpose(3, 0, 1, 2)
                # (RGB 3, frame T=16, H=224, W=224)
                in_frames = np.expand_dims(in_frames, axis=0)
                # (1, 3, 16 * segments_num, 224, 224)
                in_frames = torch.from_numpy(in_frames).float()
                # torch.Size([1, 3, 16, 224, 224])

                in_frames = in_frames.to(device)

                with torch.no_grad():
                    v_output = self.backbone(in_frames)
                    # torch.Size([1, 710])
                    v_score = self.classifier(v_output)
                    # torch.Size([1, 1])
                self.scores.append(v_score.cpu().item())

                self.v_frames = []

                if self.scores[-1] > threshold:
                    # if True:
                    anomaly_text = f"Anomaly detected, score: {self.scores[-1]}"
                    for f_step, (frame_i, results_i, temp_for_db_i) in enumerate(
                        zip(self.frame_list, self.results_list, self.tfdb_list)
                    ):
                        if results_i[0].boxes is not None:  # Check if there are results and boxes

                            # Get the boxes
                            boxes = results_i[0].boxes.xywh.cpu()

                            if results_i[0].boxes.id is not None:
                                # If 'int' attribute exists (there are detections), get the track IDs

                                track_ids = results_i[0].boxes.id.int().cpu().tolist()

                                # Loop through the detections and add data to the DataFrame
                                # anomaly_text = ""  # Initialize the anomaly text

                                # 한 프레임에서 검출된 사람만큼 돌아가는 반복문. 2명이면 각 id 별로 아래 연산들이 진행됨.
                                for i, box in zip(range(0, len(track_ids)), results_i[0].boxes.xywhn.cpu()):

                                    x, y, w, h = box
                                    keypoints = results_i[0].keypoints.xyn[i].cpu().numpy().flatten().tolist()

                                    xywhk = np.array([float(x), float(y), float(w), float(h)] + keypoints)

                                    xywhk = list(map(lambda x: str(round(x, 4)), xywhk))

                                    temp_for_db_i["bbox"][f"id {i}"] = " ".join(xywhk[:4])

                                    temp_for_db_i["keypoints"][f"id {i}"] = " ".join(xywhk[4:])

                            else:
                                # If 'int' attribute doesn't exist (no detections), set track_ids to an empty list
                                track_ids = []

                            # self.scores.append(mse_unit)

                            # Display the annotated frame
                            # cv2.imshow("YOLOv8 Tracking", annotated_frame)

                        # else:
                        # If no detections, display the original frame without annotations
                        # self.scores.append(mse_unit)
                        # cv2.imshow("YOLOv8 Tracking", frame)

                        temp_for_db_i["score"] = self.scores[-1]

                        # upload frame to s3
                        frame_url = await self.upload_frame_s3(self.s3, frame_i)

                        # upload frame, ts, bbox, kp to db
                        await self.upload_frame_db(self.db, temp_for_db_i, frame_url)

                        await self.websocket.send_text(f"{timestamp}: {anomaly_text}")

                # 초기화
                self.frame_list = []
                self.results_list = []
                self.tfdb_list = []
