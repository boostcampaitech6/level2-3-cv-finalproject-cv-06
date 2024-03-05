from datetime import timedelta, datetime, date
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from starlette import status

from db.database import get_db
from crud import crud
from schemas import schemas
from models.models import User
from core.config import get_settings

from api.user_router import get_current_user
import boto3

settings = get_settings()

upload_router = APIRouter(
    prefix="/upload",
)

s3 = boto3.client("s3",aws_access_key_id=settings.AWS_ACCESS_KEY ,aws_secret_access_key=settings.AWS_SECRET_KEY)

@upload_router.post("/", status_code=status.HTTP_204_NO_CONTENT)
async def upload(name: str = Form(...), video: UploadFile = File(...),
                 date: date = Form(...),
                 is_realtime: bool = Form(...),
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    
    user_id = current_user.user_id
    _upload_create = schemas.UploadCreate(name=name, date=date,
                                          is_realtime=is_realtime, user_id=user_id)
    crud.create_upload(db=db, upload=_upload_create)
    uploaded = crud.get_upload_id(db=db, user_id=user_id, name=name, date=date)
    video_url = f"video/{user_id}/{uploaded.upload_id}/{video.filename}"
    
    _video_create = schemas.VideoCreate(video_url=video_url, upload_id=uploaded.upload_id)
    crud.create_video(db=db, video=_video_create)
    
    s3_upload_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="video 를 s3 저장소 업로드에 실패했습니다."
    )
    try:
        s3.upload_fileobj(
            video.file,
            settings.BUCKET,
            video_url
        )
    except:
        raise s3_upload_exception
    
    return {
        "user_id": user_id,
        "email": current_user.email,
        "upload_id": uploaded.upload_id,
        "name": name,
        "date": date,
        "video_name": video.filename,
        "video_url": video_url
    }