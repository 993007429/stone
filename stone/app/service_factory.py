from stone.app.container import AppContainer
from stone.modules.slice.application.services import SliceService
from stone.modules.ai.application.services import AiService
from stone.modules.user.application.services import UserService


class AppServiceFactory(object):

    user_service: UserService = AppContainer.user_container.user_service()

    slice_service: SliceService = AppContainer.slice_container.slice_service()

    ai_service: AiService = AppContainer.ai_container.ai_service()
