import enum
import uuid
from typing import Optional, Callable, TypeVar, Any


class BaseEnum(enum.Enum):

    @classmethod
    def get_by_value(cls, value: Any):
        try:
            return cls(value.value if isinstance(value, BaseEnum) else value)
        except ValueError:
            return None

    @classmethod
    def get_by_name(cls, name):
        try:
            return cls[name.name if isinstance(name, BaseEnum) else name]
        except KeyError:
            return None

    def __eq__(self, other):
        return self.value == other or super(BaseEnum, self).__eq__(other)

    def __hash__(self):
        return hash(self.value)

    def translate(self, *args, **kwargs):
        return self.value


class GenericUUID(uuid.UUID):
    @classmethod
    def next_id(cls):
        return cls(int=uuid.uuid4().int)
