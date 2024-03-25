from datetime import datetime
import os
import uuid

from fastapi import APIRouter, Response, Request, Form, UploadFile, File, Depends, BackgroundTasks, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from utils.config import settings
from database import crud
from database import schemas
from database.database import get_db
from utils.security import get_current_user
from utils.utils import s3, run_model

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
    background_tasks: BackgroundTasks,
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

    uploaded = crud.get_upload_id(db=db, user_id=user.user_id, name=name, date=datetime)[-1]

    video_name = uuid.uuid1()

    # model inference 에서 s3 에 올릴 주소 그대로 db 에 insert
    video_url = f"video/{user.user_id}/{uploaded.upload_id}/{video_name}{file_ext}"
    _video_create = schemas.VideoCreate(video_url=video_url, upload_id=uploaded.upload_id)
    crud.create_video(db=db, video=_video_create)
    _complete_create = schemas.Complete(completed=False, upload_id=uploaded.upload_id)
    crud.create_complete(db=db, complete=_complete_create)

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
    }

    background_tasks.add_task(run_model, video_url, upload_file, info, s3, settings, db)

    redirect_url = f"/album/details?user_id={info['user_id']}&upload_id={info['upload_id']}"

    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
