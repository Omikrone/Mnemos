from pydantic import BaseModel, Field

class ChatResponse(BaseModel):
    text: str = Field(
        ...,
        description="Texte généré par le modèle"
    )
