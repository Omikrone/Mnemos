from fastapi import APIRouter
from app.models.response import ChatResponse
from app.models.request import ChatRequest
from app.llm.loader import load_llm

router = APIRouter()
llm = load_llm()

@router.post("/")
def chat(req: ChatRequest) -> ChatResponse:
    return llm.generate_text(req.prompt)
