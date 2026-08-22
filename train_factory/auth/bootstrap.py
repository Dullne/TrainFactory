"""Safe first-administrator bootstrap."""

from .user_service import user_service


def bootstrap_default_admin(settings, service=user_service) -> str:
    """Ensure an administrator exists when authentication is enabled."""
    if not settings.auth_enabled:
        return "disabled"

    if service.has_active_admin():
        return "existing"

    password = settings.default_admin_password
    if not password:
        raise RuntimeError(
            "DEFAULT_ADMIN_PASSWORD is required while authentication is "
            "enabled and no active administrator exists"
        )
    if len(password) < 10:
        raise RuntimeError(
            "DEFAULT_ADMIN_PASSWORD must contain at least 10 characters"
        )

    existing = service.get_user_by_username(settings.default_admin_username)
    if existing:
        if not service.verify_user_password(existing["user_id"], password):
            # 用户名被非 admin 用户占用：降级为告警而非阻断启动（否则单次
            # 注册即可造成服务永久无法启动）。运维需手动处理该用户或改
            # DEFAULT_ADMIN_USERNAME。
            import logging

            logging.getLogger(__name__).warning(
                "Configured administrator username '%s' is taken by a non-admin "
                "user; skipping bootstrap. Fix the user or change "
                "DEFAULT_ADMIN_USERNAME.",
                settings.default_admin_username,
            )
            return "skipped-collision"
        service.set_account_flags(
            existing["user_id"],
            is_active=True,
            is_admin=True,
            acting_user_id=None,
            bootstrap=True,
        )
        return "promoted"

    try:
        service.create_user(
            username=settings.default_admin_username,
            password=password,
            email=settings.default_admin_email,
            is_admin=True,
        )
    except Exception:
        # 多副本同时首次启动的竞态：另一个副本可能刚创建了 admin。
        # 幂等处理——重新确认后放行，避免唯一约束冲突冒泡导致启动失败。
        if service.has_active_admin():
            return "existing-race"
        raise
    return "created"
