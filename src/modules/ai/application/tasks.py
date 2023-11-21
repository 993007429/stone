from src.celery import app
from src.app.service_factory import AppServiceFactory


@app.task()
def run_ai_task(**kwargs):
    AppServiceFactory.ai_service.run_ai_task(**kwargs)
