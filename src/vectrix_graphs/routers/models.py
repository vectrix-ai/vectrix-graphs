from fastapi import APIRouter

router = APIRouter()

response = {
    "object": "list",
    "data": [
        {
            "id": "navid_ai_demo_local",
            "object": "model",
            "created": 1686935002,
            "owned_by": "vectrix",
        },
        {
            "id": "navid_ai_demo_online",
            "object": "model",
            "created": 1686935002,
            "owned_by": "vectrix",
        },
    ],
}


@router.get("/models")
async def get_models():
    return response
