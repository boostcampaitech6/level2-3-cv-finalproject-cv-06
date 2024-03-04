from typing import Union
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse


templates = Jinja2Templates(directory="templates")
app = FastAPI()

@app.get("/")
async def home(request:Request):
	return templates.TemplateResponse("main.html", {'request': request})

@app.get('/login')
async def login(request:Request):
	return templates.TemplateResponse('login.html',{'request': request})

@app.get('/mainlogin')
async def login(request:Request):
	return templates.TemplateResponse('main_login.html',{'request': request})

@app.post('/login')
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    return RedirectResponse(url='/mainlogin', status_code=303)

@app.get('/signup')
async def signup(request:Request):
	return templates.TemplateResponse('signup.html',{'request': request})

@app.get('/album')
async def album(request:Request):
	return templates.TemplateResponse('album.html',{'request': request})

@app.get('/upload')
async def upload(request:Request):
	return templates.TemplateResponse('upload.html',{'request': request})

if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port=8000)