import os
import uuid
from datetime import datetime

import requests
from database import crud, schemas
from database.database import get_db
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from utils.config import settings
from utils.security import get_current_user
from utils.utils import s3

templates = Jinja2Templates(directory="templates")

router = APIRouter(
    prefix="/upload",
)


@router.get("")
async def upload_get(request: Request):
    user = get_current_user(request)
    err_msg = {"file_ext": None}
    if not user:
        return RedirectResponse(url="/user/login")

    return templates.TemplateResponse(
        "upload.html", {"request": request, "token": user.email, "err": err_msg}
    )


@router.post("")
async def upload_post(
    request: Request,
    name: str = Form(...),
    upload_file: UploadFile = File(...),
    datetime: datetime = Form(...),
    thr: float = Form(...),
    db: Session = Depends(get_db),
):

    user = get_current_user(request)
    err_msg = {"file_ext": None}

    if not user:
        return RedirectResponse(url="/user/login")

    file_ext = os.path.splitext(upload_file.filename)[-1]
    if file_ext != ".mp4":
        err_msg["file_ext"] = "파일 형식이 다릅니다.(mp4만 지원 가능)"
        return templates.TemplateResponse(
            "upload.html", {"request": request, "token": user.email, "err": err_msg}
        )

    _upload_create = schemas.UploadCreate(
        name=name, date=datetime, is_realtime=False, thr=thr, user_id=user.user_id
    )
    crud.create_upload(db=db, upload=_upload_create)

    uploaded = crud.get_upload_id(
        db=db, user_id=user.user_id, name=name, date=datetime
    )[-1]

    video_name = uuid.uuid1()

    # model inference 에서 s3 에 올릴 주소 그대로 db 에 insert
    video_url = f"video/{user.user_id}/{uploaded.upload_id}/{video_name}{file_ext}"
    _video_create = schemas.VideoCreate(
        video_url=video_url, upload_id=uploaded.upload_id
    )
    crud.create_video(db=db, video=_video_create)
    _complete_create = schemas.Complete(completed=False, upload_id=uploaded.upload_id)
    crud.create_complete(db=db, complete=_complete_create)

    s3_upload_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="video 를 s3 저장소 업로드에 실패했습니다.",
    )

    try:
        s3.upload_fileobj(
            upload_file.file,
            settings.BUCKET,
            video_url,
            ExtraArgs={"ContentType": "video/mp4"},
        )
    except:
        raise s3_upload_exception

    info = {
        "user_id": user.user_id,
        "email": user.email,
        "upload_id": uploaded.upload_id,
        "name": name,
        "date": datetime,
        "threshold": uploaded.thr,
        "video_name": upload_file.filename,
        "video_uuid_name": video_name,
        "video_ext": file_ext,
        "video_id": crud.get_video(db=db, upload_id=uploaded.upload_id).video_id,
        "video_url": video_url,
    }

    model_data = {
        "user_id": user.user_id,
        "upload_id": uploaded.upload_id,
        "threshold": uploaded.thr,
        "video_uuid_name": str(video_name),
        "video_ext": file_ext,
        "video_id": crud.get_video(db=db, upload_id=uploaded.upload_id).video_id,
        "video_url": video_url,
    }

    model_server_url = settings.UPLOAD_MODEL_SERVER_IP
    try:
        response = requests.post(model_server_url, json=model_data)
        response.raise_for_status()  # 응답 상태 코드가 200이 아닌 경우 예외 발생
        print("Model execution started successfully.")
    except requests.RequestException:
        e = "모델 서버에서 오류가 발생했습니다."
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e)

    redirect_url = (
        f"/album/details?user_id={info['user_id']}&upload_id={info['upload_id']}"
    )

    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
