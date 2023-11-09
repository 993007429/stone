import asyncio
from apiflask import APIBlueprint

ai_blueprint = APIBlueprint('ai', __name__, url_prefix='/ai')


@ai_blueprint.get('/qwer')
async def get():
    await asyncio.sleep(2)
    return {'hel1': 'hahaha'}
