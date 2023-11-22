from src.celery.app import celery_task


@celery_task
def run_ai_task(**kwargs):
    from src.app.service_factory import AppServiceFactory
    print("celery task start...")
    AppServiceFactory.ai_service.run_ai_task(**kwargs)
