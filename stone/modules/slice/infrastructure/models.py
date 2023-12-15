from sqlalchemy import Column, String, Integer, DateTime, BigInteger, Float, Boolean, SmallInteger, text, \
    literal_column, Index, JSON

from stone.modules.slice.domain.enum import DataType, SliceAnalysisStat
from stone.seedwork.infrastructure.models import Base


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
        Index('idx_data_type', 'data_type'),
    )

    key = Column(String(255), nullable=False, unique=True, comment='切片唯一key')
    name = Column(String(255), nullable=False, server_default=text('""'), comment='切片名')
    parent_id = Column(Integer, nullable=True, comment='关联数据(父级数据ID)')
    data_type = Column(SmallInteger, nullable=False, server_default=literal_column(f'{DataType.wsi.value}'), comment='数据类型(WSI、ROI、Patch)')
    slice_num = Column(String(255), nullable=True, comment='切片号')
    entry_stat = Column(Boolean, nullable=False, server_default=literal_column('0'), comment='入库状态')
    entry_date = Column(DateTime, nullable=True, comment='入库日期')
    macro = Column(String(255), nullable=True, comment='宏观图(open slide读取macro image, 卡片视图切换)')
    analysis_stat = Column(SmallInteger, nullable=False, server_default=literal_column(f'{SliceAnalysisStat.default.value}'), comment='处理状态')
    ai_model = Column(String(255), nullable=True, comment='AI模块(最后一次处理数据所用的AI模块)')
    ai_suggest = Column(JSON, nullable=True, comment='AI建议(最后一次AI分析结果)')
    last_analysis = Column(DateTime, nullable=True, comment='AI分析日期(最后一次AI分析时间)')
    qc = Column(String(255), nullable=True, comment='质控结果(合格/不合格{不合格原因eg模糊}/null, 双击展开质控详细记录)')
    last_qc = Column(DateTime, nullable=True, comment='质控日期(最后一次质控标签时间)')
    pathology_num = Column(String(255), nullable=True, comment='病理号')
    clinical_info = Column(String(255), nullable=True, comment='临床信息')
    slice_source = Column(String(255), nullable=True, comment='切片来源')
    sample_type = Column(String(255), nullable=True, comment='样本类型')
    sample_site = Column(String(255), nullable=True, comment='取样部位')
    data_collector = Column(String(255), nullable=True, comment='数据采集人')
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


class Label(Base):
    __tablename__ = 'label'

    name = Column(String(255), nullable=False, unique=True)
    creator = Column(String(255), nullable=True)


class FilterTemplate(Base):
    __tablename__ = 'filter_template'

    name = Column(String(255), nullable=False)
    logic = Column(String(255), nullable=False)
    fields = Column(JSON, default=[], nullable=True)


class DataSet(Base):
    __tablename__ = 'dataset'

    __table_args__ = (
        Index('idx_userid', 'userid'),
    )

    name = Column(String(255), nullable=True, unique=True)
    remark = Column(String(255), nullable=True)
    creator = Column(String(255), nullable=True)
    userid = Column(BigInteger, nullable=False)


class Analysis(Base):

    __tablename__ = 'analysis'

    __table_args__ = (
        Index('idx_userid', 'userid'),
        Index('idx_results', 'slice_id', 'ai_model', 'model_version'),
    )

    key = Column(String(255), nullable=False, unique=True, comment='分析记录唯一key')
    ai_model = Column(String(255), nullable=False, comment='模型')
    model_version = Column(String(255), nullable=False, comment='模型版本')
    status = Column(Integer, nullable=False, comment='运算状态')
    time_consume = Column(Float, nullable=False, comment='运算时间')
    userid = Column(Integer, nullable=False, comment='执行用户ID')
    username = Column(String(255), nullable=False, comment='执行用户名')
    slice_id = Column(Integer, nullable=False, comment='切片ID')
    slice_key = Column(String(255), nullable=False, comment='切片key')
    ai_suggest = Column(JSON, nullable=True, comment='分析结果: AI建议')
