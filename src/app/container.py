from src.app.request_context import RequestContext, request_context
from src.modules.analysis.application.services import AnalysisService
from src.modules.analysis.domain.services import AnalysisDomainService
from src.modules.analysis.infrastructure.repositories import SQLAlchemyAnalysisRepository
from src.modules.slice.application.services import SliceService
from src.modules.slice.domain.services import SliceDomainService
from src.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository
from src.modules.user.application.services import UserService
from src.modules.user.domain.services import UserDomainService
from dependency_injector import containers, providers

from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository


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


class AnalysisContainer(containers.DeclarativeContainer):

    core_container = providers.DependenciesContainer()

    repository = providers.Factory(SQLAlchemyAnalysisRepository, session=core_container.request_context.provided.db_session)

    analysis_domain_service = providers.Factory(AnalysisDomainService, repository=repository)

    analysis_service = providers.Factory(AnalysisService, domain_service=analysis_domain_service)


class AppContainer(containers.DeclarativeContainer):

    core_container = providers.Container(CoreContainer)

    user_container = providers.Container(UserContainer, core_container=core_container)

    slice_container = providers.Container(SliceContainer, core_container=core_container)

    analysis_container = providers.Container(AnalysisContainer, core_container=core_container)
