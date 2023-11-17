from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.app.base_schema import DurationField, PageQuery
from src.modules.slice.domain.value_objects import LogicType, AiType


class SlicePageQuery(PageQuery):
    pass


class Filter(Schema):
    field = String(required=True)
    condition = String(required=True)
    value = String(required=True)


class SliceFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class SliceOut(Schema):
    slice_id = Integer(required=True)
    parent_id = String(required=True, description='关联数据(父级数据ID)')
    slice_name = String(required=True, validate=[Length(0, 255)])
    slice_num = Integer(required=True, description='切片号')
    slice_label = String(required=True, description='切片标签(openslide读取label, 卡片视图切换)')
    macro_img = String(required=True, description='宏观图(openslide读取macro image, 卡片视图切换)')
    analysis_stat = String(required=True, description='处理状态', validate=[Length(0, 255)])
    slice_type = String(required=True, description='数据类型(WSI、ROI、Patch)')
    wh_stat = String(required=True, description='入库状态', validate=[OneOf([AiType.admin.value, AiType.user.value])])
    ai_model = String(required=True, description='AI模块(最后一次处理数据所用的AI模块)')
    ai_suggest = String(required=True, description='AI建议(最后一次AI分析结果)')
    last_analysis = String(required=True, description='AI分析日期(最后一次AI分析时间)')
    qua = String(required=True, description='质控结果(合格/不合格{不合格原因eg模糊}/null, 双击展开质控详细记录)')
    last_qua = String(required=True, description='质控日期(最后一次质控标签时间)')
    pathology_num = String(required=True, description='病理号')
    clinica_info = String(required=True, description='临床信息')
    slice_source = String(required=True, description='切片来源')
    module = String(required=True, description='模块')
    sam_type = String(required=True, description='样本类型')
    sam_site = String(required=True, description='取样部位')
    data_collector = String(required=True, description='数据采集人')
    storage_date = String(required=True, description='入库日期')
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
    resolution = String(required=True, description='分辨率')
    cc = String(required=True, description='颜色校正(有/无/无法读取/不明)')
    mpp = String(required=True, description='mpp')
    f_path = String(required=True, description='存储路径')
    f_size = String(required=True, description='文件大小')
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')

    anno_stat = String(required=False, description='标注状态(待标注/已标注) 二期')
    anno_count = String(required=False, description='标注数量 二期')
    last_anno = DateTime(required=False, description='标注日期(最后一次数据标注更新时间) 二期')
    review_stat = String(required=False, description='复核状态(待复核/已复核) 二期')
    review = String(required=False, description='复核诊断结果(最后一次内部医生复核诊断结果) 二期')
    review_date = String(required=False, description='复核日期(最后一次内部医生复核诊断日期) 二期')


class SingleSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SliceOut)


class ListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(SliceOut))
