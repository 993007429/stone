from typing import Optional
from datetime import datetime

from src.seedwork.domain.entities import BaseEntity
from src.seedwork.domain.value_objects import GenericUUID


class SliceEntity(BaseEntity):
    slice_key: Optional[str]
    parent_id: Optional[int]
    name: Optional[str]
    data_type: Optional[int]
    anal_stat: Optional[int]
    wh_stat: Optional[bool]
    no: Optional[str]
    label: Optional[str]
    macro: Optional[str]
    thumbnail: Optional[str]
    ai_model: Optional[str]
    ai_suggest: Optional[str]
    last_anal: Optional[datetime]
    qua: Optional[str]
    last_qua: Optional[datetime]
    p_num: Optional[str]
    clin_info: Optional[str]
    slice_so: Optional[str]
    module: Optional[str]
    sam_type: Optional[str]
    sam_site: Optional[str]
    data_co: Optional[str]
    sto_date: Optional[datetime]
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
    f_path: Optional[str]
    f_size: Optional[float]
    created_at: Optional[datetime]
    is_deleted: Optional[bool]

    class Config:
        orm_mode = True

    @property
    def slice_path(self):
        return ''


class LabelEntity(BaseEntity):
    name: str
    count: int
    creator: str
    is_deleted: int = 0

    class Config:
        orm_mode = True










