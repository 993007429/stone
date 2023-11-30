from sqlalchemy import Column, DateTime, BigInteger, Boolean, func, literal_column
from sqlalchemy import inspect
from sqlalchemy.orm import declarative_base


class _Base:
    __table_args__ = {'mysql_engine': 'InnoDB'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    last_modified = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    is_deleted = Column(Boolean, nullable=False, server_default=literal_column('0'))

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
