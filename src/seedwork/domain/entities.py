from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BaseEntity(object):

    created_at: datetime
    last_modified: datetime
    id: int

    @property
    def dict(self):
        return self.__dict__
