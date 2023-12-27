import os
import sys
import argparse

os.environ['STONE_ENV'] = 'test'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CreateUser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='create user')
        self.parser.add_argument('-u', '--username', help='username', default='sa')
        self.parser.add_argument('-p', '--password', help='password', default='dyj123456')
        self.parser.add_argument('-r', '--role', help='admin or user', default='admin')
        self.parser.add_argument('-c', '--creator', help='creator', default='sa')

    def run(self, args):
        from stone.app.request_context import request_context
        from stone.app.service_factory import AppServiceFactory
        request_context.connect_db()
        res = AppServiceFactory.user_service.create_user(username=args.username, password=args.password, role=args.role, creator=args.creator)
        print(res.dict())
        request_context.close_db()


if __name__ == '__main__':
    command = CreateUser()
    args = command.parser.parse_args()
    command.run(args)
