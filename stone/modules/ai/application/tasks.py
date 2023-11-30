import logging

from stone.celery.app import celery_task

logger = logging.getLogger(__name__)


@celery_task
def run_ai_task(task_param):
    from stone.app.service_factory import AppServiceFactory
    logger.info('celery task start...')
    AppServiceFactory.ai_service.run_ai_task(task_param)
