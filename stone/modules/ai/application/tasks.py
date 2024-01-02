import logging

from celery.app import celery_task

logger = logging.getLogger(__name__)


@celery_task
def run_ai_task(slice_id: int, ai_model: str, model_version: str):
    from stone.app.service_factory import AppServiceFactory
    logger.info('celery task start...')
    return AppServiceFactory.ai_service.run_ai_task(slice_id, ai_model, model_version)
