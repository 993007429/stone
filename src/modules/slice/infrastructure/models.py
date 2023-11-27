from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime, BigInteger

from src.seedwork.infrastructure.models import Base


class Slice(Base):
    __tablename__ = 'slice'

    slice_key = Column(String(255), nullable=False)
    parent_id = Column(Integer, nullable=True)
    name = Column(String(255), nullable=False)
    data_type = Column(String(255), nullable=False)
