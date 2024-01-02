from functools import wraps

from celery import Celery
from kombu import Exchange, Queue

import setting

default_exchange = Exchange('default', type='direct')
host_exchange = Exchange('host', type='direct')

QUEUE_NAME_DEFAULT = 'default'
QUEUE_NAME_AI_TASK = 'ai_task'
ROUTING_KEY_DEFAULT = f'host.{QUEUE_NAME_DEFAULT}'
ROUTING_KEY_AI_TASK = f'host.{QUEUE_NAME_AI_TASK}'


def make_celery_config():
    config = dict(
        broker_url=setting.CELERY_BROKER_URL,
        result_backend=setting.CELERY_BACKEND_URL,
        broker_failover_strategy='round-robin',
        task_queue_ha_policy='all',
        broker_connection_max_retries=3,
        task_serializer='pickle',
        result_serializer='pickle',
        accept_content=['pickle', 'json'],
        result_expires=1800,
        worker_prefetch_multiplier=1,
        worker_max_memory_per_child=10000000,
        worker_send_task_events=True,
        task_send_sent_event=True,
        broker_transport_options={'visibility_timeout': 60}
    )
    return config


app = Celery('stone.celery', include=['stone.modules.ai.application.tasks'])


app.config_from_object(make_celery_config())
app.conf.update(
    BROKER_POOL_LIMIT=None,
    CELERY_QUEUES=(
        Queue(QUEUE_NAME_DEFAULT, default_exchange, routing_key='default'),
        Queue(QUEUE_NAME_AI_TASK, host_exchange, routing_key=f'host.{QUEUE_NAME_AI_TASK}'),
    ),
    CELERY_DEFAULT_QUEUE='default',
    CELERY_DEFAULT_EXCHANGE='default',
    CELERY_DEFAULT_ROUTING_KEY='default',
    CELERY_ROUTES=(
        {'stone.modules.ai.application.tasks.run_ai_task': {
            'queue': QUEUE_NAME_AI_TASK,
            'routing_key': ROUTING_KEY_AI_TASK,
        }}
    )
)


def celery_task(f):

    @wraps(f)
    def _remote_func(*args, **kwargs):
        from stone.app.request_context import request_context
        request_context.connect_db()
        result = f(*args, **kwargs)
        if result.err_code != 0:
            request_context.close_db(commit=False)
        else:
            request_context.close_db()
        return result

    func = app.task(_remote_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func.apply_async(args, kwargs)
        return result
    return wrapper


if __name__ == '__main__':

    app.start()
