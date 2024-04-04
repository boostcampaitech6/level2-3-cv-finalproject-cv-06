from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "55c84cbfa7f9e183da2179cb34cc45526bea05ee80b5bef66ed950534730bf5d"
    ALGORITHM: str = "HS256"
    # 60 minutes * 24 hours * 7 days = 7 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    MYSQL_SERVER_IP: str
    MYSQL_SERVER_PORT: int
    MYSQL_SERVER_USER: str
    MYSQL_SERVER_PASSWORD: str
    MYSQL_DATABASE: str

    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    BUCKET: str

    SMTP_ADDRESS: str
    SMTP_PORT: int
    MAIL_ACCOUNT: str
    MAIL_PASSWORD: str

    UPLOAD_MODEL_SERVER_IP: str
    STREAM_MODEL_SERVER_IP: str

    class Config:
        env_file = ".env"


settings = Settings()
