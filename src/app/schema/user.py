from apiflask import Schema, PaginationSchema
from apiflask.fields import Integer, String, List, Nested
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.modules.user.domain.value_objects import RoleType


class UserIn(Schema):
    username = String(required=True, validate=[Length(0, 100)])
    password = String(required=True, validate=[Length(8, 32)])
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])


class PageQuery(Schema):
    page = Integer(load_default=1)
    per_page = Integer(load_default=10, validate=Range(max=100))


class LoginIn(Schema):
    username = String(required=True, validate=[Length(0, 100)])
    password = String(required=True, validate=[Length(8, 32)])

