from sqlalchemy import inspect, JSON, Column, String, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    password_hash = Column(String)
    role = Column(String)

    @property
    def dict(self) -> dict:
        mapper = inspect(self.__class__)
        return {column.key: getattr(self, column.key) for column in mapper.attrs}
