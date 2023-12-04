from sqlalchemy import Column, String, text

from stone.modules.user.domain.value_objects import RoleType
from stone.seedwork.infrastructure.models import Base


class User(Base):
    __tablename__ = 'user'

    username = Column(String(255), nullable=False, server_default=text('""'))
    password_hash = Column(String(255), nullable=True)
    role = Column(String(255), nullable=False, server_default=text(f'"{RoleType.user.value}"'))
    creator = Column(String(255), nullable=True, comment='创建者')


