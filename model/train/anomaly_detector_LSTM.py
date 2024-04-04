import argparse
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from ultralytics import YOLO

# Create an argument parser
parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument(
    "--video_path",
    type=str,
    default="../videos/anomaly_1.mp4",
    help="Path to the video file",
)
parser.add_argument(
    "--threshold", type=float, default=0.02, help="Anomaly detection threshold"
)

# Parse the command-line arguments
args = parser.parse_args()


def display_text(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    cv2.putText(
        frame, text, position, font, font_scale, font_color, font_thickness, cv2.LINE_AA
    )


# Load the YOLOv8 model
model = YOLO("yolov8n-pose.pt")

# Load the LSTM autoencoder model
autoencoder_model = torch.load("model.h5")  # Replace 'model.h5' with your model's path

# Define the standard frame size (change these values as needed)
standard_width = 640
standard_height = 480

# Open the video file
if args.video_path == "0":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Define sequence_length, prediction_time, and n_features based on your model's configuration
sequence_length = 20
prediction_time = 1
n_features = 38

# Initialize a dictionary to store separate buffers for each ID
id_buffers = defaultdict(lambda: [])


# Define a function to calculate MSE between two sequences
def calculate_mse(seq1, seq2):
    return np.mean(np.power(seq1 - seq2, 2))


# Define the anomaly threshold
threshold = args.threshold  # Adjust as needed

# Loop through the video frames
frame_count = 0
net_mse = 0
avg_mse = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame_count += 1  # Increment frame count

    if success:
        frame = cv2.resize(frame, (standard_width, standard_height))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        if results[0].boxes is not None:  # Check if there are results and boxes
            # Get the boxes
            boxes = results[0].boxes.xywh.cpu()

            if results[0].boxes.id is not None:
                # If 'int' attribute exists (there are detections), get the track IDs
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Loop through the detections and add data to the DataFrame
                anomaly_text = ""  # Initialize the anomaly text
                for i, box in zip(
                    range(0, len(track_ids)), results[0].boxes.xywhn.cpu()
                ):
                    x, y, w, h = box
                    keypoints = (
                        results[0].keypoints.xyn[i].cpu().numpy().flatten().tolist()
                    )

                    # Create a dictionary with named columns for keypoints
                    keypoints_dict = {
                        f"Keypoint_{j}": float(val)
                        for j, val in enumerate(keypoints, 0)
                    }

                    # Append the keypoints to the corresponding ID's buffer
                    id_buffers[track_ids[i]].append(
                        [float(x), float(y), float(w), float(h)] + keypoints
                    )

                    # If the buffer size reaches the threshold (e.g., 20 data points), perform anomaly detection
                    if len(id_buffers[track_ids[i]]) >= 20:
                        # Convert the buffer to a NumPy array
                        buffer_array = np.array(id_buffers[track_ids[i]])

                        # Scale the data (you can use the same scaler you used during training)
                        scaler = MinMaxScaler()
                        buffer_scaled = scaler.fit_transform(buffer_array)

                        # Create sequences for prediction
                        x_pred = buffer_scaled[-sequence_length:].reshape(
                            1, sequence_length, n_features
                        )

                        # Predict the next values using the autoencoder model
                        x_pred = autoencoder_model.predict(x_pred)

                        # Inverse transform the predicted data to the original scale
                        x_pred_original = scaler.inverse_transform(
                            x_pred.reshape(-1, n_features)
                        )

                        # Calculate the MSE between the predicted and actual values
                        mse = calculate_mse(
                            buffer_array[-prediction_time:], x_pred_original
                        )
                        print(mse)
                        net_mse = mse + net_mse
                        avg_mse = net_mse / frame_count
                        # Check if the MSE exceeds the threshold to detect an anomaly
                        if mse > 1.5 * avg_mse * 0.25 + 0.75 * threshold:
                            if anomaly_text == "":
                                anomaly_text = f"Anomaly detected for ID {track_ids[i]}"
                            else:
                                anomaly_text = f"{anomaly_text}, {track_ids[i]}"
                            print(anomaly_text)

                        # Remove the oldest data point from the buffer to maintain its size
                        id_buffers[track_ids[i]].pop(0)
            else:
                anomaly_text = ""
                # If 'int' attribute doesn't exist (no detections), set track_ids to an empty list
                track_ids = []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            display_text(
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

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        else:
            # If no detections, display the original frame without annotations
            cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
