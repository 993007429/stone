from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime, BigInteger, Table, ForeignKey
from sqlalchemy.orm import relationship

from src.seedwork.infrastructure.models import Base


class SliceLabel(Base):
    __tablename__ = "slice_label"

    id = Column(BigInteger, primary_key=True)
    slice_id = Column(Integer, ForeignKey("slice.id"))
    label_id = Column(Integer, ForeignKey("label.id"))


class Slice(Base):
    __tablename__ = 'slice'

    slice_key = Column(String(255), nullable=False)
    parent_id = Column(Integer, nullable=True)
    name = Column(String(255), nullable=False)
    data_type = Column(String(255), nullable=False)
    wh_stat = Column(String(255), nullable=False)
    is_deleted = Column(Integer, nullable=False)

    labels = relationship('Label', secondary='slice_label', back_populates='slices')


class Label(Base):
    __tablename__ = 'label'

    name = Column(String(255), nullable=False)
    is_deleted = Column(Integer, nullable=False)

    slices = relationship('Slice', secondary='slice_label', back_populates='labels')











