from dataclasses import dataclass


@dataclass
class BaseEntity(object):

    # def __init__(self, **kwargs):
    #     super(BaseEntity, self).__init__()
    #     for k, v in kwargs.items():
    #         self.__setattr__(k, v)

    @property
    def dict(self):
        return self.__dict__
