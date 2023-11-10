from src.modules.user.domain.services import UserDomainService


class UserService(object):

    def __init__(self, domain_service: UserDomainService):
        # super(UserService, self).__init__()
        self.domain_service = domain_service

    def create_user(self, username: str, password: str):
        err_msg, new_user = self.domain_service.create_user(username, password)
        if err_msg:
            return 1
        return new_user
