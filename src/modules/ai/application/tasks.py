from src.celery import app


@app.task()
def run_ai_task():
    from src.app.service_factory import AppServiceFactory
    res = AppServiceFactory.ai_service.run_ai_task()
    return res
