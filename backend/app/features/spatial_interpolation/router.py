from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def test_component1():
    return {"message": "Component 1 working"}