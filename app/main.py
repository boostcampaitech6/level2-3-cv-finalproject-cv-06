from typing import Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from datetime import timedelta, datetime
from sqlalchemy.orm import Session
from starlette import status

# from api.album_router import router 
from api import user_router, album_router
# from api.video_router import video_router
# from api.upload_router import upload_router
from db.database import get_db, db_engine
from models import models
from crud.crud import pwd_context
from core.config import get_settings
from jose import jwt, JWTError

templates = Jinja2Templates(directory="templates")

app = FastAPI()

settings = get_settings()

@app.get("/")
async def main_get(request:Request):
	token = request.cookies.get("access_token", None)
	return templates.TemplateResponse("main.html", {'request': request, "token": token})

@app.post("/")
async def main_post(request: Request):
	body = await request.form()
	user_info_query = Session(db_engine).query(models.User).filter(models.User.email == body['email']).first()

	if not user_info_query or not pwd_context.verify(body['password'], user_info_query.password):
		return RedirectResponse(url="/user/login")

	data = {
        "sub": user_info_query.email,
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    }
	access_token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

	token = {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user_info_query.email
    }
	
	template_response = templates.TemplateResponse('main.html', {'request': request, 'token': token})
    # 쿠키 저장
	template_response.set_cookie(key="access_token", value=token, expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES), httponly=True)
    
	return template_response

app.include_router(user_router.router)

# @app.get('/upload')
# async def upload(request:Request):
# 	return templates.TemplateResponse('upload.html',{'request': request})

# app.include_router(upload_router.router)

app.include_router(album_router.router)


if __name__ == '__main__':
	uvicorn.run("main:app", host='0.0.0.0', port=30011, reload=True)