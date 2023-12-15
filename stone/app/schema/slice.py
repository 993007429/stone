import os
from datetime import datetime

from apiflask import Schema
from apiflask.fields import File
from marshmallow import ValidationError, validates_schema
from marshmallow.fields import Integer, String, DateTime, Float, Dict, Raw, Boolean, List, Nested
from marshmallow.validate import OneOf

import setting
from stone.app.base_schema import PageQuery, PaginationSchema, validate_positive_integers
from stone.modules.slice.infrastructure.models import Slice
from stone.seedwork.domain.enum import LogicType, Condition
from stone.utils.load_yaml import load_yaml

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

yams_path = os.path.join(setting.PROJECT_DIR, 'yams')
models_and_versions = load_yaml(os.path.join(yams_path, 'deploy.yaml'))


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


class BaseFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter()), required=True)


class SliceFilter(BaseFilter):
    label_ids = List(Integer(), required=False)


class SliceBase(Schema):
    key = String(required=True, description='切片唯一ID')
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
    slice_key = String(required=True)
    slice_filename = String(required=True)


class ApiSingleSliceUploadOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleSliceUploadOut())


class SingleSliceFieldsOut(Schema):
    fields = List(String())


class ApiSingleSliceFieldsOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleSliceFieldsOut())


class SingleSliceOut(Schema):
    slice = Nested(SliceOut())


class ApiSingleSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleSliceOut())


class ListSliceOut(Schema):
    slices = List(Nested(SliceOut()))


class ApiListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListSliceOut())
    pagination = Nested(PaginationSchema())


class SliceId(Schema):
    id: int = Integer(required=True, description='切片ID')


class SliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')


class SliceUpdateIn(SliceBase):
    ids = List(Integer(required=True), description='切片ID列表')


class SliceAndLabelIdsIn(Schema):
    slice_ids = List(Integer(required=True, validate=validate_positive_integers), description='切片ID列表', required=True)
    label_ids = List(Integer(required=True, validate=validate_positive_integers), description='标签ID列表', required=True)

    @validates_schema(pass_many=True)
    def validate_value(self, data, **kwargs):
        slice_ids = data.get('slice_ids')
        label_ids = data.get('label_ids')

        if not slice_ids or not label_ids:
            raise ValidationError('List cannot be empty.')


class QueryTileIn(Schema):
    slice_key = String(required=True, description='切片key')
    slice_name = String(required=True, description='切片name')
    x = Integer(required=True)
    y = Integer(required=True)
    z = Integer(required=True)


class ComparisonSliceFilter(BaseFilter):
    ai_model = String(required=True, description='模块')
    model_versions = List(String(), description='模型列表')

    @validates_schema(pass_many=True, pass_original=True)
    def validate_value(self, data, raw_data, **kwargs):
        ai_model = data.get('ai_model')
        model_versions = data.get('model_versions')

        if ai_model not in models_and_versions:
            raise ValidationError(f'{ai_model} is not a valid model')

        for model, versions in models_and_versions.items():
            if ai_model == model:
                versions = [version for version in versions]
                if not set(model_versions).issubset(set(versions)):
                    raise ValidationError(f'{model_versions} contains invalid version for model {ai_model}')


class AnalysisResultOut(Schema):
    analysis_id = Integer(required=True)
    ai_model = String(required=True, description='模块')
    model_version = String(required=True, description='模型')
    ai_suggest = Dict(keys=String(), values=List(String()), description='模型分析结果')


class ComparisonSliceOut(Schema):
    slice_id = Integer(required=True)
    slice_name = String(required=True, description='切片名')
    analysis_results = List(Nested(AnalysisResultOut()), description='各模型分析结果')


class ComparisonListSliceOut(Schema):
    comparison_slices = List(Nested(ComparisonSliceOut()))


class ApiComparisonListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ComparisonListSliceOut())
    pagination = Nested(PaginationSchema())
