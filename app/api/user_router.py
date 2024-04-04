from database import crud, models
from database.database import get_db
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from utils.security import pwd_context

templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix="/user")


@router.get("/signup")
async def signup_get(request: Request, db: Session = Depends(get_db)):
    err_msg = {"user": None, "pw": None, "check_pw": None}
    return templates.TemplateResponse(
        "signup.html", {"request": request, "err": err_msg}
    )


@router.post("/signup")
async def signup_post(request: Request, db: Session = Depends(get_db)):
    body = await request.form()
    user, pw, check_pw = body["email"], body["pw"], body["check_pw"]
    err_msg = {"user": None, "pw": None, "check_pw": None}

    if not user:
        err_msg["user"] = "empty email"
    elif not pw:
        err_msg["pw"] = "empty password"
    elif pw != check_pw:
        err_msg["check_pw"] = "not equal password and check_password"
    else:
        user = db.query(models.User).filter(models.User.email == body["email"]).first()

        if user:
            err_msg["user"] = "invalid email"
        else:
            user_info = models.User(email=body["email"], password=body["pw"])

            crud.create_user(db, user_info)
            return RedirectResponse(url="/user/login")

    return templates.TemplateResponse(
        "signup.html", {"request": request, "err": err_msg}
    )


@router.get("/login")
async def login_get(request: Request):
    err_msg = {"user": None, "pw": None}
    return templates.TemplateResponse(
        "login.html", {"request": request, "err": err_msg}
    )


@router.post("/login")
async def login_post(request: Request, db: Session = Depends(get_db)):
    body = await request.form()
    user, pw = body["email"], body["pw"]
    err_msg = {"user": None, "pw": None}

    if body.get("check_pw", None):
        return templates.TemplateResponse(
            "login.html", {"request": request, "err": err_msg}
        )

    if not user:
        err_msg["user"] = "empty email"
    elif not pw:
        err_msg["pw"] = "empty password"
    else:
        user = db.query(models.User).filter(models.User.email == body["email"]).first()
        if not user:
            err_msg["user"] = "invalid email"
        elif not pwd_context.verify(body["pw"], user.password):
            err_msg["pw"] = "invalid password"
        else:
            return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "login.html", {"request": request, "err": err_msg}
    )


@router.get("/logout")
async def logout_get(request: Request):
    access_token = request.cookies.get("access_token", None)
    template = RedirectResponse(url="/")
    if access_token:
        template.delete_cookie(key="access_token")
    return template
