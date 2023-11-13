from apiflask import APIFlask, Schema
from apiflask.fields import String, Integer, Field


class BaseResponseSchema(Schema):
    code = Integer()
    message = String()
    data = Field()
