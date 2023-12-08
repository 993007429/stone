from sqlalchemy import Column, String, Integer, Float, Index

from stone.seedwork.infrastructure.models import Base


class Analysis(Base):

    __tablename__ = 'analysis'

    __table_args__ = (
        Index('idx_userid', 'userid'),
        Index('idx_slice_id', 'slice_id'),
    )

    analysis_key = Column(String(255), nullable=False, comment='分析记录key')
    ai_model = Column(String(255), nullable=False, comment='模型')
    model_version = Column(String(255), nullable=False, comment='模型版本')
    status = Column(Integer, nullable=False, comment='运算状态')
    time_consume = Column(Float, nullable=False, comment='运算时间')
    userid = Column(Integer, nullable=False, comment='执行用户ID')
    username = Column(String(255), nullable=False, comment='执行用户名')
    slice_id = Column(Integer, nullable=False, comment='切片ID')
    slice_key = Column(String(255), nullable=False, comment='切片key')
