from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI()

MODEL_PATH = r"C:\Users\drand\.lmstudio\models\lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, chat_template="gemma3")

class Message(BaseModel):
    role: str
    content: str | list

class Request(BaseModel):
    messages: List[Message]
    # Добавляем параметры с валидацией (ge = greater or equal, le = less or equal)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Контролирует случайность (0.0 - детерминировано, 1.0+ - креативно)")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling: порог вероятности для токенов")

@app.post("/v1/chat/completions")
def chat(req: Request):
    # Совместимость с Pydantic V1 (.dict()) и V2 (.model_dump())
    messages = [m.model_dump() if hasattr(m, 'model_dump') else m.dict() for m in req.messages]
    
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p
    )
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)