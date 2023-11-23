import enum
from typing import Optional, List, TypeVar

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject
from src.utils.id_worker import IdWorker


@enum.unique
class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"


@enum.unique
class AIType(BaseEnum):

    @classmethod
    def get_by_value(cls, value: Optional[str]):
        if isinstance(value, AIType):
            value = value.value
        if value and (value.startswith('tct') or value.startswith('lct')):
            value = value[0:3]
        if value and value.startswith('fish'):
            value = 'fishTissue'
        if value == 'tagging':
            value = 'label'
        return super().get_by_value(value)

    human = 'human'
    human_tl = 'human_tl'
    human_bm = 'human_bm'
    label = 'label'
    np = 'np'
    er = 'er'
    pr = 'pr'
    bm = 'bm'
    tct = 'tct'
    lct = 'lct'
    dna = 'dna'
    dna_ploidy = 'dna_ploidy'
    her2 = 'her2'
    ki67 = 'ki67'
    pdl1 = 'pdl1'
    cd30 = 'cd30'
    ki67hot = 'ki67hot'
    celldet = 'celldet'
    cellseg = 'cellseg'
    fish_tissue = 'fishTissue'
    model_calibrate_tct = 'model_calibrate_tct'
    model_calibrate_lct = 'model_calibrate_lct'

    @property
    def ai_name(self) -> str:
        if self == AIType.ki67:
            return AIType.ki67hot.value
        return self.value

    @property
    def display_name(self) -> str:
        return Consts.ALGOR_DICT.get(self.value, '')

    @property
    def is_human_type(self):
        return self in [self.human, self.human_tl, self.human_bm]


A = TypeVar('A', bound=AIType)


class TaskParam(BaseValueObject):
    slice_id: int
    ai_model: str
    model_version: str
    slide_path: Optional[str] = None

    @classmethod
    def new_default_roi(cls) -> dict:
        return {
            'id': IdWorker.new_mark_id_worker().get_new_id(),
            'x': [],
            'y': []
        }


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
    ai_suggest: str
    cell_marks: List[Mark] = []
    roi_marks: List[Mark] = []
    slide_quality: Optional[int] = None
    cell_num: Optional[int] = None
    prob_dict: Optional[dict] = None
    err_msg: Optional[str] = None


