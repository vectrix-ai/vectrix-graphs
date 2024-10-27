from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .routers import chat, models
import os
from dotenv import load_dotenv

# Try to load .env file if it exists (development)
# If it doesn't exist (production), it will silently continue using OS environment variables
try:
    load_dotenv()
except Exception:
    pass


app = FastAPI(
    title="vectrix-graphs",
    description="OpenAI-compatible API for graph operations. This API implements OpenAI's API interface for drop-in compatibility.",
    version="1.0.0"
)
security = HTTPBearer()

# Add this function to verify the token
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Replace this with your actual token verification logic
    if credentials.credentials != os.environ["BEARER_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Update router includes to use authentication
app.include_router(
    chat.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)
app.include_router(
    models.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)

# Root endpoint can remain public or be protected
@app.get("/")
async def root():
    return {"message": "Welcome to Vectrix Graphs API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
