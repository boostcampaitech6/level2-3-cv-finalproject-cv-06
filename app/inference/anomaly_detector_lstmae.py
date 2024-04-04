import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, time
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from database import crud, schemas
from fastapi import HTTPException
from sklearn.preprocessing import MinMaxScaler
from starlette import status
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(os.path.join(parent_dir, "model"))
from lstmae.lstm_ae import LSTMAutoEncoder


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

        # Define sequence_length, prediction_time, and n_features
        sequence_length = 20
        prediction_time = 1
        n_features = 38
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # LSTM autoencoder
        checkpoint = torch.load(
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/model/pts/LSTM_20240324_222238_best.pth"
        )
        autoencoder_model = LSTMAutoEncoder(
            num_layers=2, hidden_size=50, n_features=n_features, device=device
        )
        autoencoder_model.load_state_dict(checkpoint["model_state_dict"])
        tracker_model.to(device)
        autoencoder_model.to(device)

        # Define the standard frame size
        standard_width = 640
        standard_height = 480

        # Open the video file
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            temp_name = f"{uuid.uuid4()}.mp4"
            self.s3.download_file(self.settings.BUCKET, self.video_url, temp_name)
            cap = cv2.VideoCapture(temp_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Store the track history
        track_history = defaultdict(lambda: [])

        # Initialize a dictionary to store separate buffers for each ID
        id_buffers = defaultdict(lambda: [])

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
        threshold = self.thr

        # Loop through the video frames
        frame_count = 0
        net_mse = 0
        avg_mse = 0
        # score graph 를 위한 score list
        scores = []

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
                mse_unit = 0

                s_timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
                datetime_object = datetime.utcfromtimestamp(s_timestamp)
                timestamp = datetime_object.strftime("%H:%M:%S")
                temp_for_db["timestamp"] = timestamp

                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = tracker_model.track(frame, persist=True)

                if results[0].boxes is not None:  # Check if there are results and boxes

                    # Get the boxes
                    boxes = results[0].boxes.xywh.cpu()

                    if results[0].boxes.id is not None:
                        # If 'int' attribute exists (there are detections), get the track IDs

                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Loop through the detections and add data to the DataFrame
                        anomaly_text = ""  # Initialize the anomaly text

                        # 한 프레임에서 검출된 사람만큼 돌아가는 반복문. 2명이면 각 id 별로 아래 연산들이 진행됨.
                        for i, box in zip(
                            range(0, len(track_ids)), results[0].boxes.xywhn.cpu()
                        ):

                            x, y, w, h = box
                            keypoints = (
                                results[0]
                                .keypoints.xyn[i]
                                .cpu()
                                .numpy()
                                .flatten()
                                .tolist()
                            )

                            # Append the keypoints to the corresponding ID's buffer
                            # bbox(4), keypoints per id(34)
                            id_buffers[track_ids[i]].append(
                                [float(x), float(y), float(w), float(h)] + keypoints
                            )

                            # If the buffer size reaches the threshold (e.g., 20 data points), perform anomaly detection
                            # track_id 별 20프레임이 쌓이면 아래 연산 진행
                            if len(id_buffers[track_ids[i]]) >= 20:
                                # Convert the buffer to a NumPy array
                                buffer_array = np.array(id_buffers[track_ids[i]])

                                # Scale the data (you can use the same scaler you used during training)
                                scaler = MinMaxScaler()
                                buffer_scaled = scaler.fit_transform(buffer_array)

                                # Create sequences for prediction
                                # x_pred: [1,20,38]
                                x_pred = buffer_scaled[-sequence_length:].reshape(
                                    1, sequence_length, n_features
                                )

                                # Predict the next values using the autoencoder model
                                x_pred = torch.tensor(x_pred, dtype=torch.float32).to(
                                    device
                                )
                                x_pred = autoencoder_model.forward(x_pred)

                                # Inverse transform the predicted data to the original scale
                                x_pred_original = scaler.inverse_transform(
                                    x_pred.cpu()
                                    .detach()
                                    .numpy()
                                    .reshape(-1, n_features)
                                )

                                # Calculate the MSE between the predicted and actual values
                                mse = calculate_mse(
                                    buffer_array[-prediction_time:], x_pred_original
                                )

                                # print(mse)

                                net_mse = mse + net_mse
                                avg_mse = net_mse / frame_count

                                mse_unit += mse

                                # Check if the MSE exceeds the threshold to detect an anomaly
                                if mse > 1.5 * avg_mse * 0.25 + 0.75 * threshold:
                                    if anomaly_text == "":
                                        anomaly_text = f"Focus ID {track_ids[i]}"
                                    else:
                                        anomaly_text = f"{anomaly_text}, {track_ids[i]}"

                                    # print(anomaly_text)

                                    temp_for_db["bbox"][f"id {i}"] = " ".join(
                                        map(
                                            lambda x: str(round(x, 4)),
                                            buffer_array[-prediction_time:][0, :4],
                                        )
                                    )

                                    temp_for_db["keypoints"][f"id {i}"] = " ".join(
                                        map(
                                            lambda x: str(round(x, 4)),
                                            buffer_array[-prediction_time:][0, 4:],
                                        )
                                    )

                                # Remove the oldest data point from the buffer to maintain its size
                                id_buffers[track_ids[i]].pop(0)

                        if temp_for_db["bbox"] != {}:

                            temp_for_db["score"] = mse_unit

                            # upload frame to s3
                            frame_url = self.upload_frame_s3(self.s3, frame)

                            # upload frame, ts, bbox, kp to db
                            self.upload_frame_db(self.db, temp_for_db, frame_url)

                    else:
                        anomaly_text = ""
                        # If 'int' attribute doesn't exist (no detections), set track_ids to an empty list
                        track_ids = []

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    self.display_text(
                        annotated_frame, anomaly_text, (10, 30)
                    )  # Display the anomaly text

                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(230, 230, 230),
                            thickness=10,
                        )

                    scores.append(mse_unit)

                    # Display the annotated frame
                    output_video.write(annotated_frame)
                    # cv2.imshow("YOLOv8 Tracking", annotated_frame)

                else:
                    # If no detections, display the original frame without annotations
                    scores.append(mse_unit)
                    output_video.write(frame)
                    # cv2.imshow("YOLOv8 Tracking", frame)

            else:
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

        try:
            os.remove(temp_name)
        except:
            pass
        os.remove(output_video_path)
