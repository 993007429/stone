from sqlalchemy import Column, String, Integer, Float

from stone.seedwork.infrastructure.models import Base


class Analysis(Base):

    __tablename__ = 'analysis'

    userid = Column(Integer, nullable=False, comment='执行用户ID')
    username = Column(String(255), nullable=False, comment='执行用户名')
    slice_id = Column(Integer, nullable=False, comment='切片名')
    ai_model = Column(String(255), nullable=False, comment='模型')
    model_version = Column(String(255), nullable=False, comment='模型版本')
    status = Column(String(255), nullable=False, comment='运算状态')
    time_consume = Column(Float, nullable=False, comment='运算时间')
