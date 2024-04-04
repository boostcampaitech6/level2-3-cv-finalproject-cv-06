import os
from datetime import datetime, timedelta

import uvicorn
from api import album_router, real_time_router, upload_router, user_router
from fastapi import FastAPI, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import jwt
from utils.config import settings
from utils.security import get_current_user

templates = Jinja2Templates(directory="templates")

app = FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), "templates", "src")
app.mount("/src", StaticFiles(directory=static_dir), name="src")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main_get(request: Request):
    user = get_current_user(request)
    if user:
        return templates.TemplateResponse(
            "main.html", {"request": request, "token": user.email}
        )
    else:
        return templates.TemplateResponse(
            "main.html", {"request": request, "token": None}
        )


@app.post("/")
async def main_post(request: Request):
    body = await request.form()
    email = body["email"]
    data = {
        "sub": email,
        "exp": datetime.utcnow()
        + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    template_response = templates.TemplateResponse(
        "main.html", {"request": request, "token": email}
    )

    # 쿠키 저장
    template_response.set_cookie(
        key="access_token",
        value=token,
        expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        httponly=True,
    )

    return template_response


app.include_router(user_router.router)
app.include_router(upload_router.router)
app.include_router(album_router.router)
app.include_router(real_time_router.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=30011, reload=True)
