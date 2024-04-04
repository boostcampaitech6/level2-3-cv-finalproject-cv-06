from datetime import datetime, timedelta
from typing import Any, Union

from database import models
from database.database import engine
from fastapi import Request
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from utils.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_current_user(request: Request):
    token = request.cookies.get("access_token", None)

    try:
        if token:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            email = payload.get("sub", None)
            session = Session(engine)
            user = session.query(models.User).filter(models.User.email == email).first()
            session.close()
            if user:
                return user
            else:
                return None
        else:
            return None
    except:
        return JWTError()
