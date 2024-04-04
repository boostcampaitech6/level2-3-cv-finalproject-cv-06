import argparse
import csv
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Feature Extraction")

parser.add_argument(
    "--root",
    type=str,
    help="root folder path",
    default="/data/ephemeral/home/datasets/UCFCrime/normal/",
)

args = parser.parse_args()

# root = args.root

root = "/data/ephemeral/home/datasets/UCFCrime/normal/"

header = [
    "Filename",
    "Frame",
    "ID",
    "X",
    "Y",
    "Width",
    "Height",
    "Keypoint_0",
    "Keypoint_1",
    "Keypoint_2",
    "Keypoint_3",
    "Keypoint_4",
    "Keypoint_5",
    "Keypoint_6",
    "Keypoint_7",
    "Keypoint_8",
    "Keypoint_9",
    "Keypoint_10",
    "Keypoint_11",
    "Keypoint_12",
    "Keypoint_13",
    "Keypoint_14",
    "Keypoint_15",
    "Keypoint_16",
    "Keypoint_17",
    "Keypoint_18",
    "Keypoint_19",
    "Keypoint_20",
    "Keypoint_21",
    "Keypoint_22",
    "Keypoint_23",
    "Keypoint_24",
    "Keypoint_25",
    "Keypoint_26",
    "Keypoint_27",
    "Keypoint_28",
    "Keypoint_29",
    "Keypoint_30",
    "Keypoint_31",
    "Keypoint_32",
    "Keypoint_33",
]

csv_root = "./UCF_csv/"

if not os.path.exists(csv_root):
    os.makedirs(csv_root)


def feat_extraction():
    # Load the YOLOv8 model
    model = YOLO("yolov8n-pose.pt")

    # Define the standard frame size (change these values as needed)
    standard_width = 320
    standard_height = 240

    folder_list = os.listdir(root)
    folder_list.sort()
    print(f"==>> folder_list: {folder_list}")

    for folder_name in folder_list:
        time_start = datetime.now()

        print(f"{folder_name} feature extracting starts")

        if not os.path.exists(csv_root + folder_name):
            os.makedirs(csv_root + folder_name)

        # with open(f"{folder_name}.csv", "w") as c_file:
        #     writer = csv.writer(c_file, delimiter=",")

        #     writer.writerow(header)

        folder_path = root + folder_name + "/"

        file_list = os.listdir(root + folder_name)
        file_list.sort()
        print(f"==>> file_list: {file_list}")

        id_count = 0

        for file_name in tqdm(file_list, total=len(file_list)):
            path = folder_path + file_name

            with open(csv_root + folder_name + "/" + f"{file_name}.csv", "w") as c_file:
                writer = csv.writer(c_file, delimiter=",")

                writer.writerow(header)

            cap = cv2.VideoCapture(path)

            # Loop through the video frames
            frame_count = 0

            # Store the track history
            track_history = defaultdict(lambda: [])

            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                frame_count += 1  # Increment frame count

                if success:
                    frame = cv2.resize(frame, (standard_width, standard_height))

                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = model.track(frame, persist=True, verbose=False)

                    if (
                        results[0].boxes is not None
                    ):  # Check if there are results and boxes
                        # Get the boxes
                        # boxes = results[0].boxes.xywh.cpu()

                        if results[0].boxes.id is not None:
                            # If 'int' attribute exists (there are detections), get the track IDs
                            track_ids = results[0].boxes.id.int().cpu().tolist()

                            for i, box in zip(
                                range(0, len(track_ids)), results[0].boxes.xywhn.cpu()
                            ):
                                keypoints = (
                                    results[0]
                                    .keypoints.xyn[i]
                                    .cpu()
                                    .numpy()
                                    .flatten()
                                    .tolist()
                                )
                                box_list = box.numpy().flatten().tolist()
                                if (
                                    type(box_list) == "float"
                                    or type(keypoints) == "float"
                                ):
                                    print(f"==>> box_list: {box_list}")
                                    print(f"==>> keypoints: {keypoints}")
                                box_and_keypoints = box_list + keypoints
                                track_history[track_ids[i]].append(
                                    [[frame_count], deepcopy(box_and_keypoints)]
                                )
                else:
                    # Break the loop if the end of the video is reached
                    break

            with open(csv_root + folder_name + "/" + f"{file_name}.csv", "a") as c_file:
                writer = csv.writer(c_file, delimiter=",")
                for key in track_history.keys():
                    for f_count, b_and_k in track_history[key]:
                        row = [file_name] + f_count + [id_count + key] + b_and_k

                        writer.writerow(row)

            id_count = id_count + len(track_history.keys())

            cap.release()

        time_end = datetime.now()
        total_time = time_end - time_start
        total_time = str(total_time).split(".")[0]

        print(f"{folder_name} feature extracting ended. Elapsed time: {total_time}")


feat_extraction()
