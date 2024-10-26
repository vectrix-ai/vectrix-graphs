from fastapi import FastAPI
from .routers import chat, models

app = FastAPI()

app.include_router(chat.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Vectrix Graphs API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)