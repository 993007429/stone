from stone.app.request_context import request_context
from stone.app.service_factory import AppServiceFactory

if __name__ == '__main__':
    request_context.connect_db()
    res = AppServiceFactory.user_service.create_user(username='sa', password='dyj123456', role='admin', creator='sa')
    request_context.close_db()
