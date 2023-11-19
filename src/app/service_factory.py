from src.app.container import AppContainer
from src.modules.slice.application.services import SliceService
from src.modules.dataset.application.services import DatasetService
from src.modules.user.application.services import UserService


class AppServiceFactory(object):

    user_service: UserService = AppContainer.user_container.user_service()

    slice_service: SliceService = AppContainer.slice_container.slice_service()

    dataset_service: DatasetService = AppContainer.dataset_container.dataset_service()
