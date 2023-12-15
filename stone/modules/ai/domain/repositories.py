from abc import ABCMeta

from stone.modules.ai.domain.entities import MarkEntity
from stone.seedwork.domain.repositories import SingleModelRepository


class MarkRepository(SingleModelRepository[MarkEntity], metaclass=ABCMeta):

    def create_mark_tables(self, ai_model: str):
        ...
