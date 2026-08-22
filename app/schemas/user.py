import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: str | None
    picture: str | None
    isActive: bool
    createdAt: datetime

    model_config = {"from_attributes": True}
