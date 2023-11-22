import time

from src.celery.app import app, celery_task


@celery_task
def run_ai_task(**kwargs):
    # from src.app.service_factory import AppServiceFactory
    # print("2")
    # res = AppServiceFactory.ai_service.run_ai_task(**kwargs)
    # time.sleep(1000)
    return 1 + 2
