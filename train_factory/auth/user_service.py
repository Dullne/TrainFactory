"""
User service for authentication operations.

Handles user registration, authentication, and management.
"""

import logging
from typing import Optional, Dict, Any

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from train_factory.core.time_utils import now_naive

from ..storage.database import get_session
from ..storage.entities.user_entity import UserDB
from .password import hash_password, password_needs_rehash, verify_password
from .jwt_handler import create_access_token

logger = logging.getLogger(__name__)

_DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$"
    "iNBZN1xTznHK798LfKmNEQ$uJgzVWI7QZ4pImROzttIe1HE/xbhO7er4iQQxT5qs80"
)


class UserService:
    """Service for user-related database operations."""

    @staticmethod
    def _validate_new_password(password: str) -> None:
        """Enforce the password policy at the service boundary."""
        if not isinstance(password, str) or len(password) < 10:
            raise ValueError("Password must be at least 10 characters")
        if len(password) > 128:
            raise ValueError("Password must be at most 128 characters")

    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        """Create a user account without issuing an access token."""
        self._validate_new_password(password)

        with get_session() as session:
            stmt = select(UserDB).where(UserDB.username == username)
            if session.exec(stmt).first():
                raise ValueError(f"Username '{username}' already exists")

            if email:
                stmt = select(UserDB).where(UserDB.email == email)
                if session.exec(stmt).first():
                    raise ValueError(f"Email '{email}' already registered")

            user = UserDB(
                username=username,
                email=email,
                hashed_password=hash_password(password),
                is_admin=is_admin,
            )
            session.add(user)
            try:
                session.commit()
                session.refresh(user)
            except IntegrityError:
                session.rollback()
                raise ValueError("Username or email already exists") from None

            logger.info(f"User created: {user.user_id} ({username})")
            return user.to_dict()

    def list_users(
        self,
        limit: int,
        offset: int,
    ) -> tuple[list[Dict[str, Any]], int]:
        """List users with stable ordering and a total count."""
        if (
            type(limit) is not int
            or type(offset) is not int
            or not 1 <= limit <= 100
            or offset < 0
        ):
            raise ValueError("Invalid pagination parameters")

        with get_session() as session:
            count_stmt = select(func.count()).select_from(UserDB)
            total = int(session.exec(count_stmt).one())
            users_stmt = (
                select(UserDB)
                .order_by(UserDB.created_at.desc(), UserDB.id.desc())
                .offset(offset)
                .limit(limit)
            )
            users = session.exec(users_stmt).all()
            return [user.to_dict() for user in users], total

    def register(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new user.

        Args:
            username: Username (must be unique)
            password: Plain text password
            email: Optional email address

        Returns:
            Dict with access_token and user info

        Raises:
            ValueError: If username or email already exists
        """
        # 保留字检查：禁止用户自注册抢占配置的管理员用户名（否则 bootstrap
        # 会因用户名冲突而无法创建 admin）。
        from ..config.settings import get_settings

        reserved = (get_settings().default_admin_username or "").strip().lower()
        if reserved and username.strip().lower() == reserved:
            raise ValueError(f"Username '{username}' is reserved")

        user = self.create_user(
            username=username,
            password=password,
            email=email,
            is_admin=False,
        )
        logger.info(f"User registered: {user['user_id']} ({username})")

        token = create_access_token(
            user["user_id"],
            user["username"],
            user["token_version"],
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user,
        }

    def has_active_admin(self) -> bool:
        """Return whether at least one active administrator exists."""
        with get_session() as session:
            stmt = select(UserDB).where(
                UserDB.is_active.is_(True),
                UserDB.is_admin.is_(True),
            )
            return session.exec(stmt).first() is not None

    def verify_user_password(self, user_id: str, password: str) -> bool:
        """Verify a password for a user, returning False when absent."""
        with get_session() as session:
            stmt = (
                select(UserDB)
                .where(UserDB.user_id == user_id)
                .with_for_update()
            )
            user = session.exec(stmt).first()
            if not user:
                return False
            password_matches = verify_password(password, user.hashed_password)
            if password_matches:
                self._rehash_password_if_needed(session, user, password)
            return password_matches

    @staticmethod
    def _rehash_password_if_needed(session, user: UserDB, password: str) -> None:
        """Upgrade a verified hash without revoking otherwise valid sessions."""
        if not password_needs_rehash(user.hashed_password):
            return

        user.hashed_password = hash_password(password)
        user.updated_at = now_naive()
        session.add(user)
        session.commit()

    def set_account_flags(
        self,
        user_id: str,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None,
        acting_user_id: Optional[str] = None,
        bootstrap: bool = False,
    ) -> Dict[str, Any]:
        """Atomically update the supplied account status flags."""
        with get_session() as session:
            locked_active_admins: Optional[list[UserDB]] = None
            user = None
            must_lock_admins_first = not bootstrap and (
                is_active is False or is_admin is False
            )
            if must_lock_admins_first:
                active_admin_stmt = (
                    select(UserDB)
                    .where(
                        UserDB.is_active.is_(True),
                        UserDB.is_admin.is_(True),
                    )
                    .order_by(UserDB.id)
                    .with_for_update()
                )
                locked_active_admins = session.exec(active_admin_stmt).all()
                user = next(
                    (
                        admin
                        for admin in locked_active_admins
                        if admin.user_id == user_id
                    ),
                    None,
                )

            if user is None:
                target_stmt = (
                    select(UserDB)
                    .where(UserDB.user_id == user_id)
                    .with_for_update()
                )
                user = session.exec(target_stmt).first()
            if not user:
                raise ValueError("User not found")

            resulting_is_active = (
                user.is_active if is_active is None else is_active
            )
            resulting_is_admin = user.is_admin if is_admin is None else is_admin

            if not bootstrap:
                if acting_user_id == user_id and is_active is False:
                    raise ValueError("Account self-disable is not allowed")

                removes_active_admin = (
                    user.is_active
                    and user.is_admin
                    and not (resulting_is_active and resulting_is_admin)
                )
                if removes_active_admin:
                    if not any(
                        admin.user_id != user_id
                        for admin in locked_active_admins or []
                    ):
                        raise ValueError(
                            "Cannot change the last active administrator"
                        )

            was_active = user.is_active
            if is_active is not None:
                user.is_active = is_active
            if is_admin is not None:
                user.is_admin = is_admin
            if was_active and not user.is_active:
                user.token_version += 1
            user.updated_at = now_naive()

            session.add(user)
            session.commit()
            session.refresh(user)
            return user.to_dict()

    def reset_password(
        self,
        user_id: str,
        new_password: str,
    ) -> Dict[str, Any]:
        """Reset a password and revoke all existing user sessions."""
        self._validate_new_password(new_password)

        with get_session() as session:
            stmt = (
                select(UserDB)
                .where(UserDB.user_id == user_id)
                .with_for_update()
            )
            user = session.exec(stmt).first()
            if not user:
                raise ValueError("User not found")

            user.hashed_password = hash_password(new_password)
            user.token_version += 1
            user.updated_at = now_naive()
            session.add(user)
            session.commit()
            session.refresh(user)
            return user.to_dict()

    def revoke_sessions(self, user_id: str) -> bool:
        """Invalidate every access token currently issued for a user."""
        with get_session() as session:
            stmt = (
                select(UserDB)
                .where(UserDB.user_id == user_id)
                .with_for_update()
            )
            user = session.exec(stmt).first()
            if not user:
                raise ValueError("User not found")

            user.token_version += 1
            user.updated_at = now_naive()
            session.add(user)
            session.commit()
            return True

    def _change_password_in_session(
        self,
        user: UserDB,
        old_password: str,
        new_password: str,
        password_matches: bool,
    ) -> bool:
        """Apply a verified password change to a locked user entity."""
        self._validate_new_password(new_password)
        if not password_matches:
            raise ValueError("Current password is incorrect")

        user.hashed_password = hash_password(new_password)
        user.token_version += 1
        user.updated_at = now_naive()
        return True

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """Change a password after verifying the current password."""
        self._validate_new_password(new_password)

        with get_session() as session:
            stmt = (
                select(UserDB)
                .where(UserDB.user_id == user_id)
                .with_for_update()
            )
            user = session.exec(stmt).first()
            if not user:
                raise ValueError("User not found")

            changed = self._change_password_in_session(
                user,
                old_password=old_password,
                new_password=new_password,
                password_matches=verify_password(
                    old_password,
                    user.hashed_password,
                ),
            )
            session.add(user)
            session.commit()
            return changed

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> Dict[str, Any]:
        """Authenticate user and return token.

        Args:
            username: Username
            password: Plain text password

        Returns:
            Dict with access_token

        Raises:
            ValueError: If credentials are invalid
        """
        with get_session() as session:
            stmt = (
                select(UserDB)
                .where(UserDB.username == username)
                .with_for_update()
            )
            user = session.exec(stmt).first()

            if not user:
                verify_password(password, _DUMMY_PASSWORD_HASH)
                raise ValueError("Invalid username or password")

            if not verify_password(password, user.hashed_password):
                raise ValueError("Invalid username or password")

            if not user.is_active:
                raise ValueError("Account is disabled")

            self._rehash_password_if_needed(session, user, password)

            logger.info(f"User authenticated: {user.user_id} ({username})")

            # Generate token
            token = create_access_token(
                user.user_id,
                user.username,
                user.token_version,
            )

            return {
                "access_token": token,
                "token_type": "bearer",
            }

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id.

        Args:
            user_id: User's unique identifier

        Returns:
            User dict or None if not found
        """
        with get_session() as session:
            stmt = select(UserDB).where(UserDB.user_id == user_id)
            user = session.exec(stmt).first()
            if user:
                return user.to_dict()
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username.

        Args:
            username: Username

        Returns:
            User dict or None if not found
        """
        with get_session() as session:
            stmt = select(UserDB).where(UserDB.username == username)
            user = session.exec(stmt).first()
            if user:
                return user.to_dict()
            return None

    def update_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """Backward-compatible alias for :meth:`change_password`."""
        return self.change_password(user_id, old_password, new_password)


# Global service instance
user_service = UserService()
