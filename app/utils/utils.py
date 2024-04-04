import boto3
from botocore.config import Config
from database import crud
from fastapi import HTTPException, status
from inference.anomaly_detector import AnomalyDetector
from utils.config import settings

# from inference.anomaly_detector_lstmae import AnomalyDetector

boto_config = Config(
    signature_version="v4",
)

s3 = boto3.client(
    "s3",
    config=boto_config,
    region_name="ap-northeast-2",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
)


def run_model(video_url, info, settings, db, s3=s3):

    model = AnomalyDetector(
        video_file=video_url, info=info, s3_client=s3, settings=settings, db=db
    )
    model.run()

    crud.update_complete_status(db=db, upload_id=info["upload_id"])
    return
