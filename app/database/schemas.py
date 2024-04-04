from datetime import datetime, time
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr, field_validator
from pydantic_core.core_schema import FieldValidationInfo


class UserBase(BaseModel):
    email: EmailStr
    is_active: Optional[bool] = True


class UserCreate(UserBase):
    password: str


# Upload post 스키마
class UploadCreate(BaseModel):
    name: str
    date: datetime
    is_realtime: Optional[bool] = None
    thr: float
    user_id: int


# Video post 스키마
class VideoCreate(BaseModel):
    video_url: str
    upload_id: int


class Video(VideoCreate):
    video_id: int
    frames: List["Frame"] = []

    class Config:
        orm_mode = True


# Frame Post 스키마
class FrameCreate(BaseModel):
    frame_url: str
    time_stamp: time
    box_kp_json: Dict
    score: float
    video_id: int


class Frame(FrameCreate):
    frame_id: int

    class Config:
        orm_mode = True


class Complete(BaseModel):
    completed: bool
    upload_id: int
