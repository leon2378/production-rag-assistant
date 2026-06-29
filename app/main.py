from fastapi import FastAPI

from app.core.config import settings
from app.api.routes_upload import router as upload_router
from app.api.routes_chat import router as chat_router


app = FastAPI(title=settings.app_name)



@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(upload_router)
app.include_router(chat_router)