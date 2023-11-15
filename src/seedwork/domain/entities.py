from dataclasses import dataclass, field


@dataclass
class BaseEntity(object):

    @property
    def dict(self):
        return self.__dict__
