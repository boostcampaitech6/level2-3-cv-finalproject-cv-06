from datetime import timedelta, datetime

from fastapi import APIRouter, Response, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status

from core.config import get_settings
from db.database import get_db, db_engine
from models import models
from crud import crud
from crud.crud import pwd_context
from schemas import schemas

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


router = APIRouter(prefix="/user")
templates = Jinja2Templates(directory="templates")

@router.get("/signup")
async def signup_get(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@router.post("/signup")
async def signup_post(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@router.get("/login")
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login_post(request: Request):
    body = await request.form() 

    with Session(db_engine) as session:
        user = session.query(models.User).filter(models.User.email == body['email']).first()
        if user:
            return RedirectResponse(url='/user/signup')
        
        if body['pw'] != body['check_pw']:
            return RedirectResponse(url='/user/signup')
        
        user_info = models.User(email = body['email'],
                                password = pwd_context.hash(body['pw']))
        
        session.add(user_info)
        session.commit()
        session.refresh(user_info)
    
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/logout")
async def logout_get(request: Request):
    access_token = request.cookies.get("access_token", None)
    
    template = templates.TemplateResponse("main.html", {"request": request})

    if access_token:
        template.delete_cookie(key="access_token")
    return template

def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    else:
        user = crud.get_user_by_email(db, email=email)
        if user is None:
            raise credentials_exception
        return user