from fastapi import APIRouter
from app.models.response import ChatResponse
from app.models.request import ChatRequest

router = APIRouter()

@router.post("/")
def chat(req: ChatRequest) -> ChatResponse:
    output = "Test response to: " + req.prompt
    return ChatResponse(text=output)
