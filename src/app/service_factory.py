from src.modules.user.application.services import UserService
from src.modules.user.domain.services import UserDomainService
from dependency_injector import containers, providers

from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository


class AIService:
    def __init__(self, user_service: UserService):
        self.user_service = user_service


class UserContainer(containers.DeclarativeContainer):

    repository = providers.Factory(SQLAlchemyUserRepository)

    user_domain_service = providers.Factory(UserDomainService, repository=repository)

    user_service = providers.Factory(UserService, domain_service=user_domain_service)


class AIContainer(containers.DeclarativeContainer):

    user_container = providers.DependenciesContainer()

    ai_service = providers.Factory(AIService, user_service=user_container.user_service)


class AppContainer(containers.DeclarativeContainer):

    user_container = providers.Container(UserContainer)

    ai_container = providers.Container(AIContainer, user_container=user_container)


class AppServiceFactory(object):

    user_service: UserService = AppContainer.user_container.user_service()

    ai_service: AIService = AppContainer.ai_container.ai_service()
