import asyncio
from flask import Blueprint

ai_blueprint = Blueprint('ai', __name__, url_prefix='/ai')


@ai_blueprint.get('/qwer')
async def get():
    await asyncio.sleep(2)
    return {'hel1': 'hahaha'}
