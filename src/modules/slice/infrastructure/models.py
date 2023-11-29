from datetime import datetime

from sqlalchemy import inspect, JSON, Column, String, Integer, DateTime, BigInteger, Table, ForeignKey, Float, Boolean, \
    SmallInteger, text, literal_column, Index

from src.modules.slice.domain.value_objects import DataType, SliceAnalysisStat
from src.seedwork.infrastructure.models import Base


class SliceLabel(Base):
    __tablename__ = "slice_label"

    __table_args__ = (
        Index('idx_slice_id', 'slice_id'),
        Index('idx_label_id', 'label_id'),
    )

    slice_id = Column(BigInteger, nullable=False)
    label_id = Column(BigInteger, nullable=False)
    label_name = Column(String(255), nullable=True)


class DataSetSlice(Base):
    __tablename__ = "dataset_slice"

    __table_args__ = (
        Index('idx_dataset_id', 'dataset_id'),
        Index('idx_slice_id', 'slice_id'),
    )

    dataset_id = Column(BigInteger, nullable=False)
    slice_id = Column(BigInteger, nullable=False)


class Slice(Base):
    __tablename__ = 'slice'

    __table_args__ = (
        Index('idx_slice_key', 'slice_key'),
        Index('idx_data_type', 'data_type'),
    )

    slice_key = Column(String(255), nullable=False, server_default=text('""'), comment='切片唯一ID')
    parent_id = Column(Integer, nullable=True, comment='关联数据(父级数据ID)')
    name = Column(String(255), nullable=False, server_default=text('""'), comment='切片名')
    data_type = Column(SmallInteger, nullable=False, server_default=literal_column(f'{DataType.wsi.value}'), comment='数据类型(WSI、ROI、Patch)')
    no = Column(String(255), nullable=True, comment='切片号')
    label = Column(String(255), nullable=True, comment='切片标签(open slide读取label, 卡片视图切换)')
    macro = Column(String(255), nullable=True, comment='宏观图(open slide读取macro image, 卡片视图切换)')
    thumbnail = Column(String(255), nullable=True, comment='切片缩略图')
    anal_stat = Column(SmallInteger, nullable=False, server_default=literal_column(f'{SliceAnalysisStat.default.value}'), comment='处理状态')
    wh_stat = Column(Boolean, nullable=False, server_default=literal_column('0'), comment='入库状态')
    ai_model = Column(String(255), nullable=True, comment='AI模块(最后一次处理数据所用的AI模块)')
    ai_suggest = Column(String(255), nullable=True, comment='AI建议(最后一次AI分析结果)')
    last_anal = Column(DateTime, nullable=True, comment='AI分析日期(最后一次AI分析时间)')
    qua = Column(String(255), nullable=True, comment='质控结果(合格/不合格{不合格原因eg模糊}/null, 双击展开质控详细记录)')
    last_qua = Column(DateTime, nullable=True, comment='质控日期(最后一次质控标签时间)')
    p_num = Column(String(255), nullable=True, comment='病理号')
    clin_info = Column(String(255), nullable=True, comment='临床信息')
    slice_so = Column(String(255), nullable=True, comment='切片来源')
    module = Column(String(255), nullable=True, comment='模块')
    sam_type = Column(String(255), nullable=True, comment='样本类型')
    sam_site = Column(String(255), nullable=True, comment='取样部位')
    data_co = Column(String(255), nullable=True, comment='数据采集人')
    sto_date = Column(DateTime, nullable=True, comment='入库日期')
    img_feat = Column(String(255), nullable=True, comment='图像特征')
    diagnosis = Column(String(255), nullable=True, comment='医院诊断(跟随模块走的结构化诊断分级结果)')
    img_c = Column(String(255), nullable=True, comment='图像色度')
    mop = Column(String(255), nullable=True, comment='制片方式')
    mod = Column(String(255), nullable=True, comment='染色方式')
    mof = Column(String(255), nullable=True, comment='试剂厂家')
    pod = Column(String(255), nullable=True, comment='染色平台')
    ant_num = Column(String(255), nullable=True, comment='抗体号')
    pc_dc = Column(String(255), nullable=True, comment='阳性对照/双染')
    icd_b = Column(String(255), nullable=True, comment='采图设备品牌')
    icd_m = Column(String(255), nullable=True, comment='采图设备型号')
    icm = Column(String(255), nullable=True, comment='采图倍率')
    resol = Column(String(255), nullable=True, comment='分辨率')
    cc = Column(String(255), nullable=True, comment='颜色校正(有/无/无法读取/不明)')
    mpp = Column(String(255), nullable=True, comment='mpp')
    f_path = Column(String(255), nullable=True, comment='存储路径')
    f_size = Column(Float, nullable=True, comment='文件大小')


class Label(Base):
    __tablename__ = 'label'

    name = Column(String(255), unique=True, nullable=False)
    creator = Column(String(255), nullable=True)


class DataSet(Base):
    __tablename__ = 'dataset'

    __table_args__ = (
        Index('idx_userid', 'userid'),
    )

    name = Column(String(255), nullable=True)
    remark = Column(String(255), nullable=True)
    count = Column(Integer, nullable=True)
    creator = Column(String(255), nullable=True)
    userid = Column(BigInteger, nullable=False)









