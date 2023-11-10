from src.infra.session import get_session
from src.modules.user.domain.entities import UserEntity
from src.modules.user.domain.repositories import UserRepository
from src.modules.user.infrastructure.models import User


class SQLAlchemyUserRepository(UserRepository):

    def __init__(self):
        self.session = get_session()

    def save(self, entity: UserEntity) -> bool:
        model = User(**entity.dict())

        self.session.begin()
        self.session.add(model)
        self.session.flush([model])
        self.session.commit()

        return True


