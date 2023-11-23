from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime, BigInteger

from src.seedwork.infrastructure.models import Base


class Slice(Base):
    __tablename__ = 'slice'

    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(255), nullable=False)
    creator = Column(String(255), nullable=False, comment='创建者')