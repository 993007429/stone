from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime, BigInteger
from sqlalchemy.orm import declarative_base
from sqlalchemy import inspect, JSON
from sqlalchemy.orm import declarative_base


class _Base:
    __table_args__ = {'mysql_engine': 'InnoDB'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    last_modified = Column(DateTime, nullable=False, default=datetime.now)

    @property
    def dict(self) -> dict:
        mapper = inspect(self.__class__)
        return {column.key: getattr(self, column.key) for column in mapper.attrs}

    def set_data(self, data: dict):
        mapper = inspect(self.__class__)
        fields = [column.key for column in mapper.attrs]
        for k, v in data.items():
            if k in fields and v != getattr(self, k):
                setattr(self, k, v)


Base = declarative_base(cls=_Base, name='BaseModel')
