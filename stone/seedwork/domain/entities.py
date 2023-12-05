from pydantic import BaseModel
from datetime import datetime


class BaseEntity(BaseModel):

    id: int = None
    created_at: datetime = None
    last_modified: datetime = None
    is_deleted: bool = None

    class Config:
        orm_mode = True
