import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, time
from io import BytesIO

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from database import crud, schemas
from fastapi import HTTPException
from sklearn.preprocessing import MinMaxScaler
from starlette import status
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(os.path.join(parent_dir, "model"))
from copy import deepcopy

import vmae

# @@ timm은 0.4.12 버전 사용 필수
from timm.models import create_model


class AnomalyDetector:
    def __init__(self, video_file, info, s3_client, settings, db):
        self.video = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": settings.BUCKET, "Key": video_file},
            ExpiresIn=3600,
        )
        # print(self.video)
        self.info = info
        self.s3 = s3_client
        self.settings = settings
        self.thr = info["threshold"]
        self.video_url = f"video/{info['user_id']}/{info['upload_id']}/{info['video_uuid_name']}{info['video_ext']}"
        self.frame_url_base = f"frame/{info['user_id']}/{info['upload_id']}/"
        self.db = db

    def display_text(self, frame, text, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green color
        font_thickness = 2
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

    def upload_frame_db(self, db, temp_for_db, frame_url):

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

    def upload_frame_s3(self, s3, frame):
        s3_upload_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Frame 을 s3 저장소 업로드에 실패했습니다.",
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

    def upload_video_s3(self, s3, video):
        s3_upload_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="video 를 s3 저장소 업로드에 실패했습니다.",
        )
        video_url = self.video_url

        video_change_codec = "./temp_video_path_change_codec.mp4"

        os.system('ffmpeg -i "%s" -vcodec libx264 "%s"' % (video, video_change_codec))

        try:
            with open(video_change_codec, "rb") as video_file:
                s3.upload_fileobj(
                    video_file,
                    self.settings.BUCKET,
                    video_url,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
        except Exception as e:
            # print(e)
            raise s3_upload_exception

        os.remove(video_change_codec)

    def upload_score_graph_s3(self, s3, scores):
        plt.plot(scores, color="red")
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
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="score Graph 를 s3 저장소 업로드에 실패했습니다.",
        )
        score_graph_name = "score_graph.png"
        score_graph_url = self.frame_url_base + score_graph_name

        try:
            s3.upload_fileobj(
                BytesIO(graph),
                self.settings.BUCKET,
                score_graph_url,
                ExtraArgs={"ContentType": "image/png"},
            )
        except:
            raise s3_upload_exception

        os.remove(save_path)

    def run(self):
        # YOLO
        tracker_model = YOLO("yolov8n-pose.pt")

        # VMAE v2
        backbone = model = create_model(
            "vit_small_patch16_224",
            img_size=224,
            pretrained=False,
            num_classes=710,
            all_frames=16,
        )

        load_dict = torch.load(
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/model/pts/vit_s_k710_dl_from_giant.pth"
        )

        backbone.load_state_dict(load_dict["module"])

        tf = A.Resize(224, 224)

        # Define sequence_length, prediction_time, and n_features
        # sequence_length = 20
        # prediction_time = 1
        # n_features = 38

        # LSTM autoencoder
        checkpoint = torch.load(
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/model/pts/MIL_20240325_202019_best_auc.pth"
        )
        classifier = vmae.MILClassifier(input_dim=710, drop_p=0.3)
        classifier.load_state_dict(checkpoint["model_state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tracker_model.to(device)
        backbone.to(device)
        backbone.eval()
        classifier.to(device)
        classifier.eval()

        # Define the standard frame size
        standard_width = 640
        standard_height = 480

        # Open the video file
        cap = cv2.VideoCapture(self.video)
        temp_name = None
        if not cap.isOpened():
            temp_name = f"{uuid.uuid4()}.mp4"
            self.s3.download_file(self.settings.BUCKET, self.video_url, temp_name)
            cap = cv2.VideoCapture(temp_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_inteval = fps // 3

        # Store the track history
        track_history = defaultdict(lambda: [])

        # Initialize a dictionary to store separate buffers for each ID
        # id_buffers = defaultdict(lambda: [])

        # HTML -> H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # video writer -> ID 별 score, bbox 가 나온 영상을 s3 에 업로드
        output_video_path = "./temp_video_path.mp4"
        output_video = cv2.VideoWriter(
            output_video_path, fourcc, fps, (standard_width, standard_height)
        )

        # Define a function to calculate MSE between two sequences
        def calculate_mse(seq1, seq2):
            return np.mean(np.power(seq1 - seq2, 2))

        # anomaly threshold (default 0.02)
        # threshold = self.thr
        threshold = 0.3

        # Loop through the video frames
        frame_count = 0
        # net_mse = 0
        # avg_mse = 0
        # score graph 를 위한 score list
        scores = []
        # vmae에 입력할 frame들을 저장할 list
        v_frames = []

        # 영상 frame 임시 저장 list
        frame_list = []
        # yolo results 임시 저장 list
        results_list = []
        # temp_for_db 임시 저장 list
        tfdb_list = []

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            frame_count += 1  # Increment frame count

            if success:
                frame = cv2.resize(frame, (standard_width, standard_height))

                temp_for_db = {
                    "timestamp": None,
                    "bbox": {},
                    "keypoints": {},
                    "score": None,
                }

                # track id (사람) 별로 mse 점수가 나오기 때문에 한 frame 에 여러 mse 점수가 나옴. 이를 frame 별 점수로 구하기 위해서 변수 설정
                # mse_unit = 0

                s_timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
                datetime_object = datetime.utcfromtimestamp(s_timestamp)
                timestamp = datetime_object.strftime("%H:%M:%S")
                temp_for_db["timestamp"] = timestamp

                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = tracker_model.track(frame, persist=True)

                frame_list.append(frame.copy())
                results_list.append(deepcopy(results))
                tfdb_list.append(deepcopy(temp_for_db))

                # 1초에 3 frame만 저장해서 vmae+MIL에 사용
                if (frame_count - 1) % frame_inteval == 0:
                    v_frame = tf(image=frame)["image"]
                    # (224, 224, 3)
                    v_frame = np.expand_dims(v_frame, axis=0)
                    # (1, 224, 224, 3)
                    v_frames.append(v_frame.copy())

                    # 16 frame이 모이면 vmae+MIL 계산
                    if len(v_frames) == 16:
                        in_frames = np.concatenate(v_frames)
                        # (16, 224, 224, 3)
                        in_frames = in_frames.transpose(3, 0, 1, 2)
                        # (RGB 3, frame T=16, H=224, W=224)
                        in_frames = np.expand_dims(in_frames, axis=0)
                        # (1, 3, 16 * segments_num, 224, 224)
                        in_frames = torch.from_numpy(in_frames).float()
                        # torch.Size([1, 3, 16, 224, 224])

                        in_frames = in_frames.to(device)

                        with torch.no_grad():
                            v_output = backbone(in_frames)
                            # torch.Size([1, 710])
                            v_score = classifier(v_output)
                            # torch.Size([1, 1])
                        scores.append(v_score.cpu().item())

                        v_frames = []

                if len(frame_list) == 16 * frame_inteval:
                    for f_step, (frame_i, results_i, temp_for_db_i) in enumerate(
                        zip(frame_list, results_list, tfdb_list)
                    ):
                        if scores[-1] > threshold:
                            anomaly_text = f"Anomaly detected, score: {scores[-1]}"
                            if (
                                results_i[0].boxes is not None
                            ):  # Check if there are results and boxes

                                # Get the boxes
                                boxes = results_i[0].boxes.xywh.cpu()

                                if results_i[0].boxes.id is not None:
                                    # If 'int' attribute exists (there are detections), get the track IDs

                                    track_ids = (
                                        results_i[0].boxes.id.int().cpu().tolist()
                                    )

                                    # Loop through the detections and add data to the DataFrame
                                    # anomaly_text = ""  # Initialize the anomaly text

                                    # vmae에서 보는 frame들만 db에 저장
                                    if f_step % frame_inteval == 0:
                                        # 한 프레임에서 검출된 사람만큼 돌아가는 반복문. 2명이면 각 id 별로 아래 연산들이 진행됨.
                                        for i, box in zip(
                                            range(0, len(track_ids)),
                                            results_i[0].boxes.xywhn.cpu(),
                                        ):

                                            x, y, w, h = box
                                            keypoints = (
                                                results_i[0]
                                                .keypoints.xyn[i]
                                                .cpu()
                                                .numpy()
                                                .flatten()
                                                .tolist()
                                            )

                                            xywhk = np.array(
                                                [float(x), float(y), float(w), float(h)]
                                                + keypoints
                                            )

                                            xywhk = list(
                                                map(lambda x: str(round(x, 4)), xywhk)
                                            )

                                            temp_for_db_i["bbox"][f"id {i}"] = " ".join(
                                                xywhk[:4]
                                            )

                                            temp_for_db_i["keypoints"][f"id {i}"] = (
                                                " ".join(xywhk[4:])
                                            )

                                else:
                                    # If 'int' attribute doesn't exist (no detections), set track_ids to an empty list
                                    track_ids = []

                                # Visualize the results on the frame
                                annotated_frame = results_i[0].plot()
                                self.display_text(
                                    annotated_frame, anomaly_text, (10, 30)
                                )  # Display the anomaly text

                                # Plot the tracks
                                # for box, track_id in zip(boxes, track_ids):
                                #     x, y, w, h = box
                                #     track = track_history[track_id]
                                #     track.append((float(x), float(y)))  # x, y center point
                                #     if len(track) > 30:  # retain 90 tracks for 90 frames
                                #         track.pop(0)

                                #     # Draw the tracking lines
                                #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                #     cv2.polylines(
                                #         annotated_frame,
                                #         [points],
                                #         isClosed=False,
                                #         color=(230, 230, 230),
                                #         thickness=10,
                                #     )

                                # Display the annotated frame
                                output_video.write(annotated_frame)
                                # cv2.imshow("YOLOv8 Tracking", annotated_frame)
                            else:
                                self.display_text(
                                    frame_i, anomaly_text, (10, 30)
                                )  # Display the anomaly text
                                output_video.write(frame_i)

                            # vmae에서 보는 frame들만 db에 저장
                            if f_step % frame_inteval == 0:
                                temp_for_db_i["score"] = scores[-1]

                                # upload frame to s3
                                frame_url = self.upload_frame_s3(self.s3, frame_i)

                                # upload frame, ts, bbox, kp to db
                                self.upload_frame_db(self.db, temp_for_db_i, frame_url)

                        else:
                            anomaly_text = ""
                            output_video.write(frame_i)
                            # cv2.imshow("YOLOv8 Tracking", frame)

                    # 16 * frame_interval개 frame 표시 후 초기화
                    frame_list = []
                    results_list = []
                    tfdb_list = []

            else:
                if len(frame_list) != 0:
                    for f in frame_list:
                        output_video.write(f)

                # Break the loop if the end of the video is reached
                break

        # Release the video capture, video writer object
        cap.release()
        output_video.release()
        # cv2.destroyAllWindows()

        # upload video to s3
        self.upload_video_s3(self.s3, output_video_path)

        # upload score graph to s3
        self.upload_score_graph_s3(self.s3, scores)
        if temp_name:
            os.remove(temp_name)
        os.remove(output_video_path)
