from abc import ABCMeta, abstractmethod

from src.modules.user.domain.entities import UserEntity


class UserRepository(metaclass=ABCMeta):

    @abstractmethod
    def save(self, entity: UserEntity) -> bool:
        ...
