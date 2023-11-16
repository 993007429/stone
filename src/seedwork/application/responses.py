from typing import Optional, Generic

from pydantic import BaseModel

from src.types import T


class AppResponse(BaseModel, Generic[T]):
    http_code: int = 200
    err_code: int = 0
    message: Optional[str] = None
    data: Optional[T] = None

    def __repr__(self):
        return f'Response(err_code={self.err_code}, message={self.message})'

    @property
    def response(self):
        return {
            'code': self.err_code,
            'message': self.message,
            'data': self.data
        }
