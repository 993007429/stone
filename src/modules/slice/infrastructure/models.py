from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime, BigInteger, Table, ForeignKey
from sqlalchemy.orm import relationship

from src.seedwork.infrastructure.models import Base


class SliceLabel(Base):
    __tablename__ = "slice_label"

    slice_id = Column(BigInteger, nullable=False)
    label_id = Column(BigInteger, nullable=False)
    label_name = Column(String(255), nullable=False)


class Slice(Base):
    __tablename__ = 'slice'

    slice_key = Column(String(255), nullable=False)
    parent_id = Column(Integer, nullable=True)
    name = Column(String(255), nullable=False)
    data_type = Column(String(255), nullable=False)
    wh_stat = Column(String(255), nullable=False)


class Label(Base):
    __tablename__ = 'label'

    name = Column(String(255), nullable=False)
    creator = Column(String(255), nullable=False)











