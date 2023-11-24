from abc import ABCMeta, abstractmethod
from typing import Optional

from src.modules.ai.domain.value_objects import AIType


class AIRepository(metaclass=ABCMeta):

    @abstractmethod
    def create_mark_tables(self, ai_type: AIType):
        ...

    @property
    @abstractmethod
    def mark_table_suffix(self) -> Optional[str]:
        ...

    @mark_table_suffix.setter
    @abstractmethod
    def mark_table_suffix(self, value):
        ...

