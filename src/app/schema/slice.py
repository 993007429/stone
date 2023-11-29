from datetime import datetime

from apiflask import Schema
from apiflask.fields import Integer, String, DateTime, URL, Float, File, Dict, Raw, Boolean, List, Nested
from apiflask.validators import Range
from apiflask.validators import Length, OneOf
from marshmallow import validates, ValidationError, validates_schema
from werkzeug.utils import secure_filename

from src.app.base_schema import DurationField, PageQuery, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType, SliceAnalysisStat, DataType, Condition
from src.modules.slice.infrastructure.models import Slice

columns_with_types = {column_name: str(column.type) for column_name, column in Slice.__table__.columns.items()}

bool_fields, float_fields, int_fields, datetime_fields, str_fields = [], [], [], [], []
for column_name, column_type in columns_with_types.items():
    if column_type in ['BOOLEAN']:
        bool_fields.append(column_name)
    elif column_type in ['FLOAT']:
        float_fields.append(column_name)
    elif column_type in ['INTEGER', 'BIGINT', 'SMALLINT']:
        int_fields.append(column_name)
    elif column_type in ['DATETIME']:
        datetime_fields.append(column_name)
    elif column_type.startswith('VARCHAR'):
        str_fields.append(column_name)


class SlicePageQuery(PageQuery):
    pass


class Filter(Schema):

    field = String(required=True, validate=OneOf(Slice.__table__.columns.keys()))
    condition = String(required=True, validate=OneOf([i.value for i in list(Condition.__members__.values())]))
    value = Raw(required=True)

    @validates_schema(pass_many=True, pass_original=True)
    def validate_value(self, data, raw_data, **kwargs):
        filed = data.get('field')
        value = data.get('value')

        if filed in bool_fields:
            if not isinstance(value, bool):
                raise ValidationError('value must be a bool')
        elif filed in float_fields:
            if not isinstance(value, float):
                raise ValidationError('value must be a float')
        elif filed in int_fields:
            if not isinstance(value, int):
                raise ValidationError('value must be an int')
        elif filed in datetime_fields:
            try:
                data['value'] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except Exception:
                raise ValidationError('value must be a datatime')
        elif filed in str_fields:
            if not isinstance(value, str):
                raise ValidationError('value must be a str')


class SliceFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class ComparisonSliceFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))
    models = List(Integer(required=True), description='模型ID列表')


class SliceBase(Schema):
    slice_key = String(required=True, description='切片唯一ID')
    parent_id = Integer(required=False, description='关联数据(父级数据ID)')
    name = String(required=False, description='切片名')
    data_type = Integer(required=False, description='数据类型(WSI、ROI、Patch)', validate=[OneOf([1, 2, 3, 4])])
    anal_stat = Integer(required=False, description='处理状态', validate=[OneOf([1, 2, 3, 4, 5])])
    wh_stat = Boolean(required=False, description='入库状态', validate=[OneOf([True, False])])
    no = String(required=False, description='切片号')
    label = String(required=False, description='切片标签(open slide读取label, 卡片视图切换)')
    macro = String(required=False, description='宏观图(open slide读取macro image, 卡片视图切换)')
    thumbnail = String(required=False, description='切片缩略图')
    ai_model = String(required=False, description='AI模块(最后一次处理数据所用的AI模块)')
    ai_suggest = String(required=False, description='AI建议(最后一次AI分析结果)')
    qua = String(required=False, description='质控结果(合格/不合格{不合格原因eg模糊}/null, 双击展开质控详细记录)')
    p_num = String(required=False, description='病理号')
    clin_info = String(required=False, description='临床信息')
    slice_so = String(required=False, description='切片来源')
    module = String(required=False, description='模块')
    sam_type = String(required=False, description='样本类型')
    sam_site = String(required=False, description='取样部位')
    data_co = String(required=False, description='数据采集人')
    img_feat = String(required=False, description='图像特征')
    diagnosis = String(required=False, description='医院诊断(跟随模块走的结构化诊断分级结果)')
    img_c = String(required=False, description='图像色度')
    mop = String(required=False, description='制片方式')
    mod = String(required=False, description='染色方式')
    mof = String(required=False, description='试剂厂家')
    pod = String(required=False, description='染色平台')
    ant_num = String(required=False, description='抗体号')
    pc_dc = String(required=False, description='阳性对照/双染')
    icd_b = String(required=False, description='采图设备品牌')
    icd_m = String(required=False, description='采图设备型号')
    icm = String(required=False, description='采图倍率')
    resol = String(required=False, description='分辨率')
    cc = String(required=False, description='颜色校正(有/无/无法读取/不明)')
    mpp = String(required=False, description='mpp')
    f_path = String(required=False, description='存储路径')
    f_size = Float(required=False, description='文件大小')

    # anno_stat = String(required=False, description='标注状态(待标注/已标注) 二期')
    # anno_count = String(required=False, description='标注数量 二期')
    # last_anno = DateTime(required=False, description='标注日期(最后一次数据标注更新时间) 二期')
    # rev_stat = String(required=False, description='复核状态(待复核/已复核) 二期')
    # rev = String(required=False, description='复核诊断结果(最后一次内部医生复核诊断结果) 二期')
    # rev_date = String(required=False, description='复核日期(最后一次内部医生复核诊断日期) 二期')


class SliceIn(SliceBase):
    sto_date = DateTime(required=False, description='入库日期')
    last_anal = DateTime(required=False, description='AI分析日期(最后一次AI分析时间)')
    last_qua = DateTime(required=False, description='质控日期(最后一次质控标签时间)')


class SliceOut(SliceBase):
    id = Integer(required=True)
    created_at = DateTime(required=False, format='%Y-%m-%d %H:%M:%S')
    sto_date = DateTime(required=False, description='入库日期', format='%Y-%m-%d %H:%M:%S')
    last_anal = DateTime(required=False, description='AI分析日期(最后一次AI分析时间)', format='%Y-%m-%d %H:%M:%S')
    last_qua = DateTime(required=False, description='质控日期(最后一次质控标签时间)', format='%Y-%m-%d %H:%M:%S')
    labels = List(String(required=True), description='标签列表')


class SliceUploadIn(Schema):
    slice_file = File()


class SingleSliceUploadOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=String(required=True))


class SingleSliceFieldOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(String(required=True)))


class ModelResultOut(Schema):
    id = Integer(required=True)
    name = String(required=True, description='模型名称')
    result = String(required=True, description='模型分析结果')


class ComparisonSliceOut(Schema):
    id = Integer(required=True)
    name = String(required=True, validate=[Length(0, 255)], description='切片名')
    models = List(Nested(ModelResultOut), description='各模型分析结果')


class SingleSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=Nested(SliceOut))


class ListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(SliceOut)))
    pagination = Nested(PaginationSchema)


class SliceId(Schema):
    id: int = Integer(required=True, description='切片ID')


class SliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')


class SliceUpdateIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')
    parent_id = String(required=False, description='关联数据(父级数据ID)')
    name = String(required=False, description='切片名')
    data_type = String(required=False, description='数据类型(WSI、ROI、Patch)')
    no = Integer(required=False, description='切片号')
    label = String(required=False, description='切片标签(open slide读取label, 卡片视图切换)')
    macro = String(required=False, description='宏观图(open slide读取macro image, 卡片视图切换)')
    thumbnail = URL(required=False, description='切片缩略图')
    anal_stat = String(required=False, description='处理状态')
    wh_stat = String(required=False, description='入库状态')
    ai_model = String(required=False, description='AI模块(最后一次处理数据所用的AI模块)')
    ai_suggest = String(required=False, description='AI建议(最后一次AI分析结果)')
    last_anal = String(required=False, description='AI分析日期(最后一次AI分析时间)')
    qua = String(required=False, description='质控结果(合格/不合格{不合格原因eg模糊}/null, 双击展开质控详细记录)')
    last_qua = String(required=False, description='质控日期(最后一次质控标签时间)')
    p_num = String(required=False, description='病理号')
    clin_info = String(required=False, description='临床信息')
    slice_so = String(required=False, description='切片来源')
    module = String(required=False, description='模块')
    sam_type = String(required=False, description='样本类型')
    sam_site = String(required=False, description='取样部位')
    data_co = String(required=False, description='数据采集人')
    sto_date = String(required=False, description='入库日期')
    img_feat = String(required=False, description='图像特征')
    diagnosis = String(required=False, description='医院诊断(跟随模块走的结构化诊断分级结果)')
    img_c = String(required=False, description='图像色度')
    mop = String(required=False, description='制片方式')
    mod = String(required=False, description='染色方式')
    mof = String(required=False, description='试剂厂家')
    pod = String(required=False, description='染色平台')
    ant_num = String(required=False, description='抗体号')
    pc_dc = String(required=False, description='阳性对照/双染')
    icd_b = String(required=False, description='采图设备品牌')
    icd_m = String(required=False, description='采图设备型号')
    icm = String(required=False, description='采图倍率')
    resol = String(required=False, description='分辨率')
    cc = String(required=False, description='颜色校正(有/无/无法读取/不明)')
    mpp = String(required=False, description='mpp')
    f_path = String(required=False, description='存储路径')
    f_size = String(required=False, description='文件大小')
    created_at = DateTime(required=False, format='%Y-%m-%d %H:%M:%S')
    is_deleted = Integer(required=False, description='逻辑删除')

    anno_stat = String(required=False, description='标注状态(待标注/已标注) 二期')
    anno_count = String(required=False, description='标注数量 二期')
    last_anno = DateTime(required=False, description='标注日期(最后一次数据标注更新时间) 二期')
    rev_stat = String(required=False, description='复核状态(待复核/已复核) 二期')
    rev = String(required=False, description='复核诊断结果(最后一次内部医生复核诊断结果) 二期')
    rev_date = String(required=False, description='复核日期(最后一次内部医生复核诊断日期) 二期')


class SliceIdsOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=Integer(required=True), description='受影响切片数量')


class SliceAndLabelIdsIn(Schema):
    slice_ids = List(Integer(required=True), description='切片ID列表')
    label_ids = List(Integer(required=True, description='标签ID列表'))


class DSSliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')
    dataset_id = Integer(required=True, description='数据集ID')


class ComparisonListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(ComparisonSliceOut)))
    pagination = Nested(PaginationSchema)


class WSIIn(Schema):
    id: int = Integer(required=True, description='切片ID')
    x: int = Integer(required=True)
    y: int = Integer(required=True)
    z: int = Integer(required=True)


class ROIIn(Schema):
    id: int = Integer(required=True, description='切片ID')












