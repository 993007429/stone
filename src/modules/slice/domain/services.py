from typing import Tuple, Optional

from src.modules.slice.domain.entities import SliceEntity
from src.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository


class SliceDomainService(object):

    def __init__(self, repository: SQLAlchemySliceRepository):
        self.repository = repository

    def get_slice_by_id(self, slice_id: int) -> Tuple[Optional[SliceEntity], str]:
        slide = self.repository.get_slice_by_id(slice_id)
        if not slide:
            return None, 'no slice'
        return slide, 'get slice success'
