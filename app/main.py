from typing import Union
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates

from api.album import album_router
from api.user import user_router
from api.video import video_router
from api.upload import upload_router


templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.include_router(user_router.signup_router)
app.include_router(user_router.login_router)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup")
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
async def login(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/upload")
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


app.include_router(upload_router.router)


@app.get("/album")
async def album(request: Request):
    return templates.TemplateResponse("album.html", {"request": request})


@app.get("/album/album_page")
async def album_page(request: Request):
    return templates.TemplateResponse("album_page.html", {"request": request})


app.include_router(album_router.router)


@app.get("/logout")
async def logout(request: Request, response: Response):
    access_token = request.cookies.get("access_token")

    # 쿠키 삭제
    template_response = templates.TemplateResponse("main.html", {"request": request})
    template_response.delete_cookie(key="access_token")

    return template_response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30305)
