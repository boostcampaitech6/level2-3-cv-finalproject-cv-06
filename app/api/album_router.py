from fastapi import Request, APIRouter, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from starlette import status
import ast

from db.database import get_db, db_engine
from crud import crud
from schemas import schemas
from models import models


router = APIRouter(
    prefix="/album",
)
templates = Jinja2Templates(directory='templates')


@router.get("")
async def album_list(request: Request):
    # 쿠키의 value를 가져와서
    # 유효한 클라이언트 -> 업로드 리스트 보여주기
    # 유효하지 않은 클라이언트 -> root page 다시 띄우기
    token = request.cookies.get("access_token", None)

    if token:
        token = ast.literal_eval(token)
    else:
        return RedirectResponse(url='/')
    
    # upload list를 DB에서 가져오기
    # 1) user 테이블에서 user_id를 찾는다.
    # 2) upload 테이블에서 user_id를 가지고 업로드 된 영상을 전부 찾는다.
    # 3) upload_list에 쿼리를 담아서 album.html에 context로 보낸다.
    with Session(db_engine) as session:
        find_user_query = session.query(models.User).filter(models.User.email == token['email']).first()     # 토큰의 email을 가지고 user 테이블에서 해당 유저 쿼리를 가져온다.
        upload_list = session.query(models.Upload).filter(models.Upload.user_id == find_user_query.user_id).all()    # 클라이언트의 user_id를 기반으로 upload 테이블에서 업로드 된 정보들을 전부 찾는다.

    
    return templates.TemplateResponse('album.html', {'request': request, 'upload_list': upload_list})

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