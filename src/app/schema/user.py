from apiflask import Schema, PaginationSchema
from apiflask.fields import Integer, String, List, Nested
from apiflask.validators import Range
from apiflask.validators import Length, OneOf


class UserIn(Schema):
    username = String(required=True, validate=[Length(0, 10)])
    password = String(required=True, validate=[Length(0, 10)])


class UserOut(Schema):
    id = Integer()
    username = String()


class UsersOut(Schema):
    users = List(Nested(UserOut, only=('id', 'username')))
    pagination = Nested(PaginationSchema)


class PageQuery(Schema):
    page = Integer(load_default=1)
    per_page = Integer(load_default=10, validate=Range(max=100))
