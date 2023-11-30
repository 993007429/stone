from stone.app.request_context import RequestContext, request_context
from stone.modules.ai.application.services import AiService
from stone.modules.ai.domain.services import AiDomainService
from stone.modules.ai.infrastructure.repositories import SQLAlchemyAIRepository
from stone.modules.slice.application.services import SliceService
from stone.modules.slice.domain.services import SliceDomainService
from stone.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository
from stone.modules.user.application.services import UserService
from stone.modules.user.domain.services import UserDomainService
from dependency_injector import containers, providers

from stone.modules.user.infrastructure.repositories import SQLAlchemyUserRepository


class CoreContainer(containers.DeclarativeContainer):

    request_context = providers.Factory(RequestContext)


class UserContainer(containers.DeclarativeContainer):

    core_container = providers.DependenciesContainer()

    repository = providers.Factory(SQLAlchemyUserRepository, session=core_container.request_context.provided.db_session)

    user_domain_service = providers.Factory(UserDomainService, repository=repository)

    user_service = providers.Factory(UserService, domain_service=user_domain_service)


class SliceContainer(containers.DeclarativeContainer):

    core_container = providers.DependenciesContainer()

    repository = providers.Factory(
        SQLAlchemySliceRepository,
        session=core_container.request_context.provided.db_session
    )

    slice_domain_service = providers.Factory(SliceDomainService, repository=repository)

    slice_service = providers.Factory(SliceService, domain_service=slice_domain_service)


class AiContainer(containers.DeclarativeContainer):

    core_container = providers.DependenciesContainer()

    slice_container = providers.DependenciesContainer()

    _manual_repository = providers.Factory(
        SQLAlchemyAIRepository,
        session=core_container.request_context.provided.db_session,
        slice_db_session=core_container.request_context.provided.slice_db_session
    )

    repository = providers.Factory(
        SQLAlchemyAIRepository,
        session=core_container.request_context.provided.db_session,
        slice_db_session=core_container.request_context.provided.slice_db_session,
        manual=_manual_repository
    )

    ai_domain_service = providers.Factory(AiDomainService, repository=repository)

    ai_service = providers.Factory(
        AiService,
        domain_service=ai_domain_service,
        slice_service=slice_container.slice_service
    )


class AppContainer(containers.DeclarativeContainer):

    core_container = providers.Container(CoreContainer)

    user_container = providers.Container(UserContainer, core_container=core_container)

    slice_container = providers.Container(SliceContainer, core_container=core_container)

    ai_container = providers.Container(AiContainer, core_container=core_container, slice_container=slice_container)
