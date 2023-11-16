import time

import bcrypt
import jwt

import setting
from src.modules.user.domain.entities import UserEntity


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_token_for_user(user: UserEntity) -> str:
    expired_at = int(time.time()) + setting.JWT_EXPIRE
    payload = {"userid": user.id, "exp": expired_at}
    return jwt.encode(payload, setting.SECRET_KEY, "HS256").decode('utf-8')
