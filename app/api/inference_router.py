import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pytz
from cap_from_youtube import cap_from_youtube
from fastapi import BackgroundTasks, Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from database import crud
from database.database import get_db
from inference.rt_anomaly_detector_lstmae import RT_AnomalyDetector
from utils.config import settings
from utils.utils import run_model, s3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInfo(BaseModel):
    user_id: int
    upload_id: int
    threshold: float
    video_uuid_name: str
    video_ext: str
    video_id: int
    video_url: str


@app.get("/")
def root():
    return {"message": "모델이 돌아가용"}


# 녹화영상용
@app.post("/run_model")
async def run_model_endpoint(
    info: ModelInfo, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):

    info = info.dict()

    def run_model_task():
        run_model(info["video_url"], info, settings, db)

    background_tasks.add_task(run_model_task)

    return {"message": "Model execution started."}


# 메일을 보내야하는지 판단하는 함수
async def check_and_send_email(db, video_id, user_id, last_point, smtp):
    global last_emailed_time

    frames = crud.get_frames_with_highest_score(db=db, video_id=video_id)
    frame_timestamps = [frame.time_stamp.strftime("%H:%M:%S") for frame in frames]

    if len(frame_timestamps) < 6:
        return False

    last = datetime.strptime(frame_timestamps[-2], "%H:%M:%S")
    check = datetime.strptime(frame_timestamps[-6], "%H:%M:%S")

    if (last - check) == timedelta(seconds=4):  # 연속적으로 5초간 지속되면
        if not check <= last_point <= last:
            crud.send_email(
                db, frame_timestamps[-6], frame_timestamps[-2], user_id, smtp
            )
            last_emailed_time = last


# 과연 웹 서버와 실시간을 분리하는 것이 더 빠른가?
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):

    await websocket.accept()
    smtp = await crud.create_smtp_server()

    try:
        video_info_str = await websocket.receive_text()
        print("Received video info:", video_info_str)
        video_info = json.loads(video_info_str)
        global detector, last_emailed_time
        if detector is None:
            detector = RT_AnomalyDetector(video_info, s3, settings, db, websocket)
            detector.ready()

        if video_info["video_url"] == "web":
            while True:
                timestamp = datetime.now(pytz.timezone("Asia/Seoul"))
                # Receive bytes from the websocket
                bytes = await websocket.receive_bytes()
                data = np.frombuffer(bytes, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                await detector.run(frame, timestamp)
                await check_and_send_email(
                    db=db,
                    video_id=video_info["video_id"],
                    user_id=video_info["user_id"],
                    last_point=last_emailed_time,
                    smtp=smtp,
                )

        else:
            if "youtube" in video_info["video_url"]:
                cap = cap_from_youtube(video_info["video_url"], "240p")

            else:
                cap = cv2.VideoCapture(video_info["video_url"])

            while True:
                success, frame = cap.read()
                if not success:
                    await websocket.send_text(f"카메라 연결에 실패했습니다.")
                    break
                else:
                    timestamp = datetime.now(pytz.timezone("Asia/Seoul"))
                    await detector.run(frame, timestamp)
                    await check_and_send_email(
                        db=db,
                        video_id=video_info["video_id"],
                        user_id=video_info["user_id"],
                        last_point=last_emailed_time,
                        smtp=smtp,
                    )

                    ret, buffer = cv2.imencode(".jpg", frame)
                    await websocket.send_bytes(buffer.tobytes())

                await asyncio.sleep(0.042)

    except WebSocketDisconnect:
        await websocket.close()
        await smtp.quit()

    except Exception as e:
        # 예외 발생 시 로그 기록 및 연결 종료
        print(f"WebSocket error: {e}")
        await websocket.close()
        await smtp.quit()

    finally:
        try:
            detector.upload_score_graph_s3()
        except:
            pass
        detector = None
