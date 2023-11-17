from src.app.container import AppContainer, AIService
from src.modules.slice.application.services import SliceService
from src.modules.user.application.services import UserService


class AppServiceFactory(object):

    user_service: UserService = AppContainer.user_container.user_service()

    slice_service: SliceService = AppContainer.slice_container.slice_service()

    ai_service: AIService = AppContainer.ai_container.ai_service()
