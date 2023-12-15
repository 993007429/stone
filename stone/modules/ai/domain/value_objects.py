from typing import Optional, List

from stone.seedwork.domain.value_objects import BaseValueObject


class Mark(BaseValueObject):
    id: Optional[int] = None
    position: Optional[dict] = None
    ai_result: Optional[dict] = None
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    mark_type: Optional[int] = None
    diagnosis: Optional[dict] = None
    radius: Optional[float] = None
    area_id: Optional[int] = None
    editable: Optional[int] = None
    group_id: Optional[int] = None
    method: Optional[str] = None
    is_export: Optional[int] = None


class ALGResult(BaseValueObject):
    ai_suggest: Optional[dict]
    cell_marks: List[Mark] = []
    roi_marks: List[Mark] = []
    slide_quality: Optional[int] = None
    cell_num: Optional[int] = None
    prob_dict: Optional[dict] = None
    err_msg: Optional[str] = None
