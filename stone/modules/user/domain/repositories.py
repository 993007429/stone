from abc import ABCMeta
from typing import Optional, List, Union, Tuple

from stone.modules.user.domain.entities import UserEntity
from stone.seedwork.domain.repositories import SingleModelRepository


class UserRepository(SingleModelRepository, metaclass=ABCMeta):

    def get_user_by_name(self, username: str) -> Optional[UserEntity]:
        ...

    def get_users(self, page: int, per_page: int, names_to_exclude: Union[list, set]) -> Tuple[List[UserEntity], dict]:
        ...
