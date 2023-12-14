from typing import Optional, List
from datetime import datetime

from stone.seedwork.domain.entities import BaseEntity


class SliceEntity(BaseEntity):
    key: Optional[str]
    name: Optional[str]
    parent_id: Optional[int]
    data_type: Optional[int]
    slice_num: Optional[str]
    entry_stat: Optional[bool]
    entry_date: Optional[datetime]
    label: Optional[str]
    macro: Optional[str]
    analysis_stat: Optional[str]
    ai_model: Optional[str]
    ai_suggest: Optional[str]
    last_analysis: Optional[datetime]
    qc: Optional[str]
    last_qc: Optional[datetime]
    pathology_num: Optional[str]
    clinical_info: Optional[str]
    slice_source: Optional[str]
    sample_type: Optional[str]
    sample_site: Optional[str]
    data_collector: Optional[str]
    img_feat: Optional[str]
    diagnosis: Optional[str]
    img_c: Optional[str]
    mop: Optional[str]
    mod: Optional[str]
    mof: Optional[str]
    pod: Optional[str]
    ant_num: Optional[str]
    pc_dc: Optional[str]
    icd_b: Optional[str]
    icd_m: Optional[str]
    icm: Optional[str]
    resol: Optional[str]
    cc: Optional[str]
    mpp: Optional[str]
    created_at: Optional[datetime]

    class Config:
        orm_mode = True


class LabelEntity(BaseEntity):
    name: Optional[str]
    creator: Optional[str]

    class Config:
        orm_mode = True


class FilterTemplateEntity(BaseEntity):
    name: Optional[str]
    logic: Optional[str]
    fields: List[dict]

    class Config:
        orm_mode = True


class SliceLabelEntity(BaseEntity):
    slice_id: Optional[int]
    label_id: Optional[int]
    label_name: Optional[str]

    class Config:
        orm_mode = True


class DataSetEntity(BaseEntity):
    userid: int
    name: Optional[str]
    remark: Optional[str]
    creator: Optional[str]

    class Config:
        orm_mode = True


class DataSetSliceEntity(BaseEntity):
    dataset_id: Optional[int]
    slice_id: Optional[int]

    class Config:
        orm_mode = True
