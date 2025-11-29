from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class ReplyResponse(BaseModel):
    reply: str