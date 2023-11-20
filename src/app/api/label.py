import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.label import LabelPageQuery, LabelFilter, ListLabelOut, SingleLabelOut, LabelIn
from src.app.service_factory import AppServiceFactory

label_blueprint = APIBlueprint('标签', __name__, url_prefix='/labels')


@label_blueprint.get('')
@label_blueprint.input(LabelPageQuery, location='query')
@label_blueprint.input(LabelFilter, location='json')
@label_blueprint.output(ListLabelOut)
@label_blueprint.doc(summary='标签列表', security='ApiAuth')
def get_labels(query_data, json_data):
    res = AppServiceFactory.slice_service.get_labels(**query_data, **json_data)
    return res.response


@label_blueprint.get('/<int:label_id>')
@label_blueprint.output(SingleLabelOut)
@label_blueprint.doc(summary='标签详情', security='ApiAuth')
def get_label(slice_id):
    res = AppServiceFactory.slice_service.get_label(slice_id)
    return res.response


@label_blueprint.put('/<int:label_id>')
@label_blueprint.input(LabelIn, location='json')
@label_blueprint.output(SingleLabelOut)
@label_blueprint.doc(summary='更新标签', security='ApiAuth')
def update_label(json_data):
    res = AppServiceFactory.slice_service.update_label(**json_data)
    return res.response


@label_blueprint.delete('/<int:label_id>')
@label_blueprint.output(SingleLabelOut)
@label_blueprint.doc(summary='删除标签', security='ApiAuth')
def delete_label(label_id):
    res = AppServiceFactory.slice_service.delete_label(label_id)
    return res.response













