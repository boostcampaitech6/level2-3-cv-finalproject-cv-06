from typing import Optional

from database import crud, models
from database.database import get_db
from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from utils.config import settings
from utils.security import get_current_user
from utils.utils import s3

templates = Jinja2Templates(directory="templates")

router = APIRouter(
    prefix="/album",
)


@router.get("")
async def upload_get(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/user/login")

    album_list = crud.get_uploads(db=db, user_id=user.user_id)
    print(album_list[0].completes[0].completed)
    return templates.TemplateResponse(
        "album_list.html",
        {"request": request, "token": user.email, "album_list": album_list},
    )


@router.post("")
async def modify_name(
    request: Request,
    check_code: str = Form(...),
    upload_id: Optional[int] = Form(...),
    origin_name: Optional[str] = Form(None),
    new_name: Optional[str] = Form(None),
    is_real_time: Optional[bool] = Form(None),
    db: Session = Depends(get_db),
):
    user = get_current_user(request)

    if check_code == "edit":
        upload_info = (
            db.query(models.Upload)
            .filter(
                (models.Upload.name == origin_name)
                & (models.Upload.upload_id == upload_id)
            )
            .first()
        )
        upload_info.name = new_name

        db.add(upload_info)
        db.commit()
        db.refresh(upload_info)
    elif check_code == "delete":
        upload_info = crud.get_upload(db, upload_id)
        if upload_info:
            db.delete(upload_info)

        db.commit()
    album_list = crud.get_uploads(db=db, user_id=user.user_id)

    return templates.TemplateResponse(
        "album_list.html",
        {"request": request, "token": user.email, "album_list": album_list},
    )


@router.get("/details")
async def upload_get_one(
    request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    db: Session = Depends(get_db),
):

    user = get_current_user(request)

    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "date": None,
        "upload_name": None,
        "is_realtime": None,
        "video_id": None,
        "video_url": None,
        "frame_urls": None,
        "score_url": None,
        "complete": None,
    }

    video = crud.get_video(db=db, upload_id=upload_id)
    video_info["video_id"] = video.video_id
    uploaded = crud.get_upload(db=db, upload_id=video.upload_id)
    video_info["upload_name"] = uploaded.name
    video_info["is_realtime"] = uploaded.is_realtime
    video_info["date"] = uploaded.date.strftime("%Y-%m-%d %H:%M:%S")

    # frames = crud.get_frames(db=db, video_id=video.video_id)
    frames = crud.get_frames_with_highest_score(db=db, video_id=video.video_id)
    frame_ids = [frame.frame_id for frame in frames]
    frame_urls = [frame.frame_url for frame in frames]
    frame_timestamps = [frame.time_stamp for frame in frames]
    frame_objs = []

    video_obj = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.BUCKET, "Key": video.video_url},
        ExpiresIn=3600,
    )

    video_info["video_url"] = video_obj
    video_info["complete"] = crud.get_complete(db=db, upload_id=upload_id).completed
    if not video_info["complete"]:
        return templates.TemplateResponse(
            "album_detail.html",
            {
                "request": request,
                "token": user.email,
                "video_info": video_info,
                "loading": True,
            },
        )

    if frame_ids != []:
        for frame_id, frame_url, frame_timestamp in zip(
            frame_ids, frame_urls, frame_timestamps
        ):
            frame_obj = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.BUCKET, "Key": frame_url},
                ExpiresIn=3600,
            )
            frame_objs.append(
                (frame_id, frame_obj, frame_timestamp.strftime("%H:%M:%S"))
            )

        score_graph_url = "/".join(frame_urls[0].split("/")[:-1]) + "/score_graph.png"
        score_obj = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.BUCKET, "Key": score_graph_url},
            ExpiresIn=3600,
        )

        video_info["frame_urls"] = frame_objs
        video_info["score_url"] = score_obj

    # print(video_info)
    return templates.TemplateResponse(
        "album_detail.html",
        {
            "request": request,
            "token": user.email,
            "video_info": video_info,
            "loading": False,
        },
    )


@router.get("/details/images")
async def image_get(
    request: Request, frame_id: int = Query(...), db: Session = Depends(get_db)
):

    user = get_current_user(request)
    frame = crud.get_frame(db=db, frame_id=frame_id)
    frame_obj = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.BUCKET, "Key": frame.frame_url},
        ExpiresIn=3600,
    )
    print(frame_obj)
    print(frame.box_kp_json)
    frame_info = {
        "frame_url": frame_obj,
        "time_stamp": frame.time_stamp,
        "frame_json": frame.box_kp_json,
    }

    return templates.TemplateResponse(
        "frame.html",
        {"request": request, "token": user.email, "frame_info": frame_info},
    )
