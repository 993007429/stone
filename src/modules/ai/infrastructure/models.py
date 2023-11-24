from typing import Type

from sqlalchemy import Column, Integer, Text, Float, String, JSON

from src.seedwork.infrastructure.models import Base


class Analysis(Base):

    __tablename__ = 'analysis'

