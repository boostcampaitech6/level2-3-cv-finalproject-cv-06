import asyncio
import json
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pytz
import websockets
from cap_from_youtube import cap_from_youtube
from database import crud, schemas
from database.database import get_db
from fastapi import (
    APIRouter,
    Depends,
    Form,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# from inference.rt_anomaly_detector import RT_AnomalyDetector
from inference.rt_anomaly_detector_lstmae import RT_AnomalyDetector
from sqlalchemy.orm import Session
from utils.config import settings
from utils.security import get_current_user
from utils.utils import s3
from websockets.exceptions import ConnectionClosed

templates = Jinja2Templates(directory="templates")
router = APIRouter(
    prefix="/real_time",
)

detector = None
last_emailed_time = datetime.strptime("0:00:00", "%H:%M:%S")


@router.get("")
async def real_time_get(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/user/login")

    return templates.TemplateResponse(
        "real_time.html", {"request": request, "token": user.email}
    )


@router.post("")
async def realtime_post(
    request: Request,
    name: str = Form(...),
    real_time_video: str = Form(...),
    datetime: datetime = Form(...),
    thr: float = Form(...),
    db: Session = Depends(get_db),
):

    user = get_current_user(request)
    user = crud.get_user_by_email(db=db, email=user.email)

    # Form 과 user_id 를 이용하여 upload row insert
    _upload_create = schemas.UploadCreate(
        name=name, date=datetime, is_realtime=True, thr=thr, user_id=user.user_id
    )
    crud.create_upload(db=db, upload=_upload_create)

    # 지금 업로드된 id 획득, 클라이언트로부터 작성된 실시간 스트리밍 영상 url 획득
    uploaded = crud.get_upload_id(
        db=db, user_id=user.user_id, name=name, date=datetime
    )[-1]

    # db 에는 실시간임을 알 수 있게만 함
    video_url = f"{real_time_video}"
    _video_create = schemas.VideoCreate(
        video_url=video_url, upload_id=uploaded.upload_id
    )
    crud.create_video(db=db, video=_video_create)
    _complete_create = schemas.Complete(completed=True, upload_id=uploaded.upload_id)
    crud.create_complete(db=db, complete=_complete_create)

    # model inference 에서 사용할 정보
    info = {
        "user_id": user.user_id,
        "email": user.email,
        "upload_id": uploaded.upload_id,
        "name": uploaded.name,
        "date": uploaded.date,
        "threshold": uploaded.thr,
        "video_url": video_url,
        "video_id": crud.get_video(db=db, upload_id=uploaded.upload_id).video_id,
    }

    redirect_url = (
        f"/real_time/stream?user_id={info['user_id']}&upload_id={info['upload_id']}"
    )

    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/stream")
async def get_stream(
    request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    db: Session = Depends(get_db),
):

    user = get_current_user(request)

    video = crud.get_video(db=db, upload_id=upload_id)
    uploaded = crud.get_upload(db=db, upload_id=video.upload_id)

    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "date": uploaded.date.strftime("%Y-%m-%d %H:%M:%S"),
        "upload_name": uploaded.name,
        "thr": uploaded.thr,
        "video_id": video.video_id,
        "video_url": video.video_url,
        "is_realtime": True,
        "model_server_ip": settings.STREAM_MODEL_SERVER_IP,
    }

    # video_info = json.dumps(video_info)

    return templates.TemplateResponse(
        "stream.html",
        {"request": request, "token": user.email, "video_info": video_info},
    )


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


@router.websocket("/ws")
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


@router.get("/stream")
async def get_stream(
    request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    db: Session = Depends(get_db),
):

    user = get_current_user(request)

    video = crud.get_video(db=db, upload_id=upload_id)
    uploaded = crud.get_upload(db=db, upload_id=video.upload_id)

    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "date": uploaded.date.strftime("%Y-%m-%d %H:%M:%S"),
        "upload_name": uploaded.name,
        "thr": uploaded.thr,
        "video_id": video.video_id,
        "video_url": video.video_url,
        "is_realtime": True,
    }

    # video_info = json.dumps(video_info)

    return templates.TemplateResponse(
        "stream.html",
        {"request": request, "token": user.email, "video_info": video_info},
    )


# db 에서 실시간에서 저장되는 frame url 불러오는 코드
def fetch_data(db, upload_id):

    video = crud.get_video(db=db, upload_id=upload_id)
    frames = crud.get_frames_with_highest_score(db=db, video_id=video.video_id)
    frame_ids = [frame.frame_id for frame in frames]
    frame_urls = [frame.frame_url for frame in frames]
    frame_timestamps = [frame.time_stamp for frame in frames]
    frame_objs = []

    for frame_id, frame_url, frame_timestamp in zip(
        frame_ids, frame_urls, frame_timestamps
    ):
        frame_obj = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.BUCKET, "Key": frame_url},
            ExpiresIn=3600,
        )
        frame_objs.append((frame_id, frame_obj, frame_timestamp.strftime("%H:%M:%S")))

    return {"frame_urls": frame_objs}


@router.get("/fetch_data")
async def fetch_frame_data(upload_id: int = Query(...), db: Session = Depends(get_db)):
    frame_data = fetch_data(db, upload_id)
    return frame_data
