import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

import setting
from src.utils.encoding import StoneJsonEncoder


def json_serializer(val):
    return json.dumps(val, cls=StoneJsonEncoder, ensure_ascii=False)


def json_deserializer(val):
    return json.loads(val) if val else None


_engine = create_engine(
    setting.SQLALCHEMY_DATABASE_URI,
    json_serializer=json_serializer,
    json_deserializer=json_deserializer,
    pool_recycle=3600,
    echo=False
)

_Session = sessionmaker(
    bind=_engine,
    autocommit=False,
    autoflush=True,
    expire_on_commit=False
)


def get_session() -> Session:
    return _Session()


def get_session_by_db_uri(uri: str):
    engine = create_engine(
        uri, json_serializer=json_serializer, json_deserializer=json_deserializer, pool_recycle=300, echo=False)
    return Session(autocommit=False, autoflush=True, expire_on_commit=False, bind=engine)
