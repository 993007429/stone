from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, URL, Float, File
from apiflask.validators import Range
from apiflask.validators import Length, OneOf
from werkzeug.utils import secure_filename

from src.app.base_schema import DurationField, PageQuery, Filter, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType


class SlicePageQuery(PageQuery):
    pass


class SliceFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class ComparisonSliceFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))
    models = List(Integer(required=True), description='模型ID列表')


class SliceBase(Schema):
    slice_key = String(required=True, description='切片唯一ID')
    parent_id = String(required=True, description='关联数据(父级数据ID)')
    name = String(required=True, validate=[Length(0, 255)], description='切片名')
    data_type = String(required=True, description='数据类型(WSI、ROI、Patch)')
    no = Integer(required=True, description='切片号')
    label = String(required=True, description='切片标签(open slide读取label, 卡片视图切换)')
    macro = String(required=True, description='宏观图(open slide读取macro image, 卡片视图切换)')
    thumbnail = URL(required=True, description='切片缩略图')
    anal_stat = String(required=True, description='处理状态', validate=[Length(0, 255)])
    wh_stat = String(required=True, description='入库状态')
    ai_model = String(required=True, description='AI模块(最后一次处理数据所用的AI模块)')
    ai_suggest = String(required=True, description='AI建议(最后一次AI分析结果)')
    last_anal = String(required=True, description='AI分析日期(最后一次AI分析时间)')
    qua = String(required=True, description='质控结果(合格/不合格{不合格原因eg模糊}/null, 双击展开质控详细记录)')
    last_qua = String(required=True, description='质控日期(最后一次质控标签时间)')
    p_num = String(required=True, description='病理号')
    clin_info = String(required=True, description='临床信息')
    slice_so = String(required=True, description='切片来源')
    module = String(required=True, description='模块')
    sam_type = String(required=True, description='样本类型')
    sam_site = String(required=True, description='取样部位')
    data_co = String(required=True, description='数据采集人')
    sto_date = String(required=True, description='入库日期')
    img_feat = String(required=True, description='图像特征')
    diagnosis = String(required=True, description='医院诊断(跟随模块走的结构化诊断分级结果)')
    img_c = String(required=True, description='图像色度')
    mop = String(required=True, description='制片方式')
    mod = String(required=True, description='染色方式')
    mof = String(required=True, description='试剂厂家')
    pod = String(required=True, description='染色平台')
    ant_num = String(required=True, description='抗体号')
    pc_dc = String(required=True, description='阳性对照/双染')
    icd_b = String(required=True, description='采图设备品牌')
    icd_m = String(required=True, description='采图设备型号')
    icm = String(required=True, description='采图倍率')
    resol = String(required=True, description='分辨率')
    cc = String(required=True, description='颜色校正(有/无/无法读取/不明)')
    mpp = String(required=True, description='mpp')
    f_path = String(required=True, description='存储路径')
    f_size = String(required=True, description='文件大小')
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    is_deleted = Integer(required=True, description='逻辑删除')

    anno_stat = String(required=False, description='标注状态(待标注/已标注) 二期')
    anno_count = String(required=False, description='标注数量 二期')
    last_anno = DateTime(required=False, description='标注日期(最后一次数据标注更新时间) 二期')
    rev_stat = String(required=False, description='复核状态(待复核/已复核) 二期')
    rev = String(required=False, description='复核诊断结果(最后一次内部医生复核诊断结果) 二期')
    rev_date = String(required=False, description='复核日期(最后一次内部医生复核诊断日期) 二期')


class SliceIn(SliceBase):
    pass


class SliceInT(Schema):
    slice_key = String(required=True, description='切片唯一ID')
    parent_id = Integer(required=False, description='关联数据(父级数据ID)')
    name = String(required=True, description='切片名')
    data_type = String(required=True, description='数据类型(WSI、ROI、Patch)')


class SliceUploadIn(Schema):
    slice_file = File()


class SingleSliceUploadOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested({'slice_key': String(required=True)})


class SliceOut(SliceBase):
    id = Integer(required=True)


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
    data = Nested(SliceOut)


class ListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(SliceOut))
    pagination = Nested(PaginationSchema)


class SliceId(Schema):
    id: int = Integer(required=True, description='切片ID')


class SliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')


class SliceIdsOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Integer(required=True), description='切片ID列表')


class LabelSliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')
    label_id = Integer(required=True, description='标签ID')


class DSSliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')
    ds_id = Integer(required=True, description='数据集ID')


class ComparisonListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(ComparisonSliceOut))
    pagination = Nested(PaginationSchema)


class WSIIn(Schema):
    id: int = Integer(required=True, description='切片ID')
    x: int = Integer(required=True)
    y: int = Integer(required=True)
    z: int = Integer(required=True)


class ROIIn(Schema):
    id: int = Integer(required=True, description='切片ID')












