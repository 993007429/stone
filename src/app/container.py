from src.app.request_context import RequestContext, request_context
from src.modules.slice.application.services import SliceService
from src.modules.slice.domain.services import SliceDomainService
from src.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository
from src.modules.user.application.services import UserService
from src.modules.user.domain.services import UserDomainService
from dependency_injector import containers, providers

from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository


class AIService:

    def __init__(self, user_service: UserService):
        self.user_service = user_service


class CoreContainer(containers.DeclarativeContainer):

    request_context = providers.Factory(RequestContext)


class UserContainer(containers.DeclarativeContainer):

    core_container = providers.DependenciesContainer()

    repository = providers.Factory(SQLAlchemyUserRepository, session=core_container.request_context.provided.db_session)

    user_domain_service = providers.Factory(UserDomainService, repository=repository)

    user_service = providers.Factory(UserService, domain_service=user_domain_service)


class SliceContainer(containers.DeclarativeContainer):

    core_container = providers.DependenciesContainer()

    repository = providers.Factory(SQLAlchemySliceRepository, session=core_container.request_context.provided.db_session)

    slice_domain_service = providers.Factory(SliceDomainService, repository=repository)

    slice_service = providers.Factory(SliceService, domain_service=slice_domain_service)


class AIContainer(containers.DeclarativeContainer):

    user_container = providers.DependenciesContainer()

    ai_service = providers.Factory(AIService, user_service=user_container.user_service)


class AppContainer(containers.DeclarativeContainer):

    core_container = providers.Container(CoreContainer)

    user_container = providers.Container(UserContainer, core_container=core_container)

    slice_container = providers.Container(SliceContainer, core_container=core_container)

    ai_container = providers.Container(AIContainer, user_container=user_container)
