from fastapi import APIRouter, Depends, HTTPException, Form, Request
from sqlalchemy.orm import Session
from starlette import status

from db.database import get_db
from crud import crud
from schemas import schemas
from models.models import User

from api.user_router import get_current_user

from fastapi.templating import Jinja2Templates

router = APIRouter(
    prefix="/album",
)

templates = Jinja2Templates(directory="templates")


@router.get("/")
def album_list(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user_id = current_user.user_id
    upload_list = crud.get_uploads(db=db, user_id=user_id)

    return {"user_id": user_id, "total": len(upload_list), "upload_list": upload_list}


# @router.post("/{upload_id}")
@router.post("/album_page")
def set_interval(request: Request, start: int = Form(...), end: int = Form(...)):

    print(start, end)

    context = {}

    context["request"] = request
    context["srcinterval"] = (
        f"http://0.0.0.0:30305/testitems/C_3_13_30_BU_SYA_10-06_15-11-55_CD_RGB_DF2_M3.mp4#t={start},{end}"
    )
    print(f"==>> context['srcinterval']: {context['srcinterval']}")

    return templates.TemplateResponse("album_page.html", context)


# @router.get("/detail/{upload_id}", response_model=schemas.Upload)
# def question_detail(upload_id: int, db: Session = Depends(get_db)):
#     question = question_crud.get_question(db, question_id=question_id)
#     return question


# @router.post("/create", status_code=status.HTTP_204_NO_CONTENT)
# def question_create(_question_create: question_schema.QuestionCreate,
#                     db: Session = Depends(get_db),
#                     current_user: User = Depends(get_current_user)):
#     question_crud.create_question(db=db, question_create=_question_create,
#                                   user=current_user)


# @router.put("/update", status_code=status.HTTP_204_NO_CONTENT)
# def question_update(_question_update: question_schema.QuestionUpdate,
#                     db: Session = Depends(get_db),
#                     current_user: User = Depends(get_current_user)):
#     db_question = question_crud.get_question(db, question_id=_question_update.question_id)
#     if not db_question:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
#                             detail="데이터를 찾을수 없습니다.")
#     if current_user.id != db_question.user.id:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
#                             detail="수정 권한이 없습니다.")
#     question_crud.update_question(db=db, db_question=db_question,
#                                   question_update=_question_update)


# @router.delete("/delete", status_code=status.HTTP_204_NO_CONTENT)
# def question_delete(_question_delete: question_schema.QuestionDelete,
#                     db: Session = Depends(get_db),
#                     current_user: User = Depends(get_current_user)):
#     db_question = question_crud.get_question(db, question_id=_question_delete.question_id)
#     if not db_question:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
#                             detail="데이터를 찾을수 없습니다.")
#     if current_user.id != db_question.user.id:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
#                             detail="삭제 권한이 없습니다.")
#     question_crud.delete_question(db=db, db_question=db_question)


# @router.post("/vote", status_code=status.HTTP_204_NO_CONTENT)
# def question_vote(_question_vote: question_schema.QuestionVote,
#                   db: Session = Depends(get_db),
#                   current_user: User = Depends(get_current_user)):
#     db_question = question_crud.get_question(db, question_id=_question_vote.question_id)
#     if not db_question:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
#                             detail="데이터를 찾을수 없습니다.")
#     question_crud.vote_question(db, db_question=db_question, db_user=current_user)


# # async examples
# @router.get("/async_list")
# async def async_question_list(db: Session = Depends(get_async_db)):
#     result = await question_crud.get_async_question_list(db)
#     return result


# @router.post("/async_create", status_code=status.HTTP_204_NO_CONTENT)
# async def async_question_create(_question_create: question_schema.QuestionCreate,
#                                 db: Session = Depends(get_async_db)):
#     await question_crud.async_create_question(db, question_create=_question_create)
