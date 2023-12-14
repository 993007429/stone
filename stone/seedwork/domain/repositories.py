from abc import ABCMeta
from typing import Generic, Optional, Union, List, Tuple

from stone.seedwork.domain.entities import E
from stone.seedwork.domain.enum import LogicType


class SingleModelRepository(Generic[E], metaclass=ABCMeta):

    def save(self, entity: E) -> Optional[E]:
        ...

    def batch_save(self, entities: List[E]) -> bool:
        ...

    def get(self, pk: int) -> Optional[E]:
        ...

    def gets(self, ids: Optional[Union[list, set]]) -> List[Optional[E]]:
        ...

    def update(self, pk: int, update_data: dict) -> int:
        ...

    def batch_update(self, ids: Union[list, set], update_data: dict) -> int:
        ...

    def delete(self, pk: int) -> int:
        ...

    def batch_delete(self, ids: Union[list, set]) -> int:
        ...

    def filter(self, page: int, per_page: int, filters: list, logic: str = LogicType.and_.value, ids: Union[list, set] = None) -> Tuple[List[E], dict]:
        ...
