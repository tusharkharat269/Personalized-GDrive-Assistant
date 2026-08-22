from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def healthCheck():
    return {"status": "healthy"}
