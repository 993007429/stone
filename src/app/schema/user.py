from apiflask import Schema, PaginationSchema
from apiflask.fields import Integer, String, List, Nested
from apiflask.validators import Range
from apiflask.validators import Length, OneOf


class UserIn(Schema):
    username = String(required=True, validate=[Length(0, 10)])
    password = String(required=True, validate=[Length(0, 10)])


class UserOut(Schema):
    username = String(required=True, validate=[Length(0, 10)])


class PetQuery(Schema):
    page = Integer(load_default=1)
    per_page = Integer(load_default=20, validate=Range(max=30))


class PetOut(Schema):
    id = Integer()
    name = String()
    category = String()


class PetsOut(Schema):
    pets = List(Nested(PetOut))
    pagination = Nested(PaginationSchema)
