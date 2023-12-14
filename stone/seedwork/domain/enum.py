from stone.seedwork.domain.value_objects import BaseEnum


class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"


class Condition(BaseEnum):
    equal = 'equal'
    unequal = 'unequal'
    greater_than = 'greater_than'
    less_than = 'less_than'
    is_null = 'is_null'
    not_null = 'not_null'
    contain = 'contain'
    not_contain = 'not_contain'
