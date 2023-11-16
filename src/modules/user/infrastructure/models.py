from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime

from src.seedwork.infrastructure.mdoels import Base


class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    creator = Column(String, nullable=False)
