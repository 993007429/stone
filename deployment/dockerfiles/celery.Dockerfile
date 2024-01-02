FROM python:3.8-slim

WORKDIR /

RUN pip install --no-cache-dir celery==5.2.7

COPY ./celery_app ./celery

CMD celery -A celery.celery_app.app worker -l info
