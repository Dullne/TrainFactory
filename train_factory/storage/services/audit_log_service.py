"""
Audit log service for recording user operations.

Provides functions to create and query audit logs.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete
from sqlmodel import select, func

from train_factory.core.time_utils import now_naive

from ..database import get_session
from ..entities.audit_log_entity import AuditLogDB

logger = logging.getLogger(__name__)

# Sensitive fields to redact from request bodies
SENSITIVE_FIELDS = {'password', 'token', 'secret', 'api_key', 'apikey', 'access_token', 'refresh_token'}


class AuditLogService:
    """Service for audit log operations."""

    def _redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from data dictionary."""
        if not data:
            return data

        redacted = {}
        for key, value in data.items():
            if key.lower() in SENSITIVE_FIELDS:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_sensitive_data(value)
            else:
                redacted[key] = value
        return redacted

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        method: str,
        endpoint: str,
        status_code: int,
        resource_id: Optional[str] = None,
        username: Optional[str] = None,
        request_body: Optional[Dict[str, Any]] = None,
        response_summary: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create an audit log entry.

        Args:
            user_id: User ID performing the action
            action: Action type (see AuditAction)
            resource_type: Resource type (see AuditResource)
            method: HTTP method
            endpoint: API endpoint path
            status_code: HTTP response status code
            resource_id: ID of the affected resource
            username: Username for display
            request_body: Request body (sensitive data will be redacted)
            response_summary: Brief response summary
            ip_address: Client IP address
            user_agent: User agent string
            extra_data: Additional context

        Returns:
            Log ID if successful, None otherwise
        """
        try:
            # Redact sensitive data from request body
            safe_request_body = self._redact_sensitive_data(request_body) if request_body else None

            with get_session() as session:
                log_entry = AuditLogDB(
                    user_id=user_id,
                    username=username,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    method=method,
                    endpoint=endpoint,
                    request_body=safe_request_body,
                    status_code=status_code,
                    response_summary=response_summary,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    extra_data=extra_data,
                )
                session.add(log_entry)
                session.commit()
                session.refresh(log_entry)
                return log_entry.log_id

        except Exception as e:
            # Don't fail the main request if audit logging fails
            logger.error(f"Failed to create audit log: {e}")
            return None

    def query_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Query audit logs with filters.

        Args:
            user_id: Filter by user ID
            action: Filter by action type
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            start_time: Filter logs after this time
            end_time: Filter logs before this time
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            Tuple of (logs, total_count)
        """
        with get_session() as session:
            # Build filter conditions
            conditions = []
            if user_id:
                conditions.append(AuditLogDB.user_id == user_id)
            if action:
                conditions.append(AuditLogDB.action == action)
            if resource_type:
                conditions.append(AuditLogDB.resource_type == resource_type)
            if resource_id:
                conditions.append(AuditLogDB.resource_id == resource_id)
            if start_time:
                conditions.append(AuditLogDB.created_at >= start_time)
            if end_time:
                conditions.append(AuditLogDB.created_at <= end_time)

            # Get total count
            count_stmt = select(func.count()).select_from(AuditLogDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            statement = select(AuditLogDB)
            for cond in conditions:
                statement = statement.where(cond)
            statement = statement.order_by(AuditLogDB.created_at.desc())
            statement = statement.offset(offset).limit(limit)
            logs = session.exec(statement).all()

            return [self._log_to_dict(log) for log in logs], total

    def _log_to_dict(self, log: AuditLogDB) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        return {
            "log_id": log.log_id,
            "user_id": log.user_id,
            "username": log.username,
            "action": log.action,
            "resource_type": log.resource_type,
            "resource_id": log.resource_id,
            "method": log.method,
            "endpoint": log.endpoint,
            "request_body": log.request_body,
            "status_code": log.status_code,
            "response_summary": log.response_summary,
            "ip_address": log.ip_address,
            "user_agent": log.user_agent,
            "extra_data": log.extra_data,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }

    def get_user_activity_summary(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get activity summary for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Activity summary with action counts
        """
        start_time = now_naive() - timedelta(days=days)

        with get_session() as session:
            # Count by action type
            statement = select(
                AuditLogDB.action,
                func.count(AuditLogDB.id).label("count")
            ).where(
                AuditLogDB.user_id == user_id,
                AuditLogDB.created_at >= start_time
            ).group_by(AuditLogDB.action)

            results = session.exec(statement).all()

            action_counts = {row.action: row.count for row in results}

            return {
                "user_id": user_id,
                "period_days": days,
                "action_counts": action_counts,
                "total_actions": sum(action_counts.values()),
            }

    def cleanup_old_logs(
        self,
        retention_days: int = 90,
        batch_size: int = 1000,
    ) -> int:
        """
        Remove audit logs older than retention period.

        Args:
            retention_days: Number of days to retain logs

        Returns:
            Number of logs deleted
        """
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        cutoff_time = now_naive() - timedelta(days=retention_days)
        deleted_count = 0

        with get_session() as session:
            while True:
                expired_ids = list(
                    session.exec(
                        select(AuditLogDB.id)
                        .where(AuditLogDB.created_at < cutoff_time)
                        .order_by(AuditLogDB.id)
                        .limit(batch_size)
                    ).all()
                )
                if not expired_ids:
                    break

                session.exec(
                    delete(AuditLogDB).where(AuditLogDB.id.in_(expired_ids))
                )
                session.commit()
                deleted_count += len(expired_ids)

        logger.info(
            "Cleaned up %s audit logs older than %s days",
            deleted_count,
            retention_days,
        )
        return deleted_count


# Global service instance
audit_log_service = AuditLogService()
