"""Model authorization, separate from permission to connect to an HTTP host."""

import posixpath
from typing import Any, Dict, Optional, Set

import httpx
from fastapi import HTTPException
from sqlmodel import select

from ...config.settings import get_settings
from ..database import get_session
from ..entities.deployment_entity import DeploymentDB


def canonical_model_endpoint(endpoint: str, provider: str = "xinference") -> str:
    """Compare the same normalized API base that HTTP clients will contact."""
    from .model_config_service import normalize_api_endpoint_url

    if not endpoint:
        return ""
    try:
        parsed = httpx.URL(normalize_api_endpoint_url(endpoint, provider))
    except (ValueError, httpx.InvalidURL) as exc:
        raise HTTPException(status_code=400, detail="Invalid model API endpoint") from exc
    path = posixpath.normpath("/" + parsed.path.lstrip("/")).rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return str(parsed.copy_with(host=parsed.host.rstrip("."), path=path))


def owned_model_uids_for_endpoint(
    endpoint: str, user_id: Optional[str], provider: str = "xinference",
) -> Set[str]:
    return registered_shared_models_for_endpoint(endpoint, user_id, provider)[1]


def registered_shared_models_for_endpoint(
    endpoint: str, user_id: Optional[str], provider: str = "",
) -> tuple[bool, Set[str]]:
    """Recognize all registered deployments without trusting request labels.

    Inspect only identity/lifecycle fields, never deployment credentials. Both
    internal endpoints and the addresses generated for model configs identify
    the same resource. A stopped resource stays protected but is not usable.
    """
    from ...deployment.deployment_service import deployment_service
    from ..entities.deployment_replica_entity import DeploymentReplicaDB

    canonical_endpoint = canonical_model_endpoint(endpoint, provider)
    registered = False
    owned: Set[str] = set()
    # Hostname filtering alone misses the HOST_IP aliases emitted by deployment
    # config creation. Do not impose an arbitrary limit on the identity catalog.
    statement = select(
        DeploymentDB.xinference_endpoint, DeploymentDB.user_id,
        DeploymentDB.model_uid, DeploymentDB.status,
        DeploymentDB.deploy_mode, DeploymentDB.port, DeploymentDB.deployment_id,
        DeploymentDB.config["replica_schema_version"].as_integer(),
        DeploymentDB.started_at,
        DeploymentDB.config["runtime_managed"].as_boolean(),
        DeploymentDB.config["read_only"].as_boolean(),
    )
    with get_session() as session:
        rows = session.exec(statement).all()
        replica_deployments = {
            row[6]: row for row in rows if row[4] == "container" and row[7] == 1
        }
        replica_rows = []
        if replica_deployments:
            replica_rows = session.exec(select(
                DeploymentReplicaDB.endpoint, DeploymentReplicaDB.deployment_id,
                DeploymentReplicaDB.status, DeploymentReplicaDB.health_status,
            ).where(DeploymentReplicaDB.deployment_id.in_(replica_deployments))).all()

    # Host discovery can involve OS/DNS I/O. Resolve it at most once per check,
    # and only when comparing an address that can actually be a generated alias.
    host_ip = None
    aliases: Dict[tuple[str, Optional[int]], Set[str]] = {}

    def matches_endpoint(stored_endpoint: str, container_port: Optional[int] = None) -> bool:
        nonlocal host_ip
        try:
            canonical = canonical_model_endpoint(stored_endpoint)
            if canonical == canonical_endpoint:
                return True
            key = (canonical, container_port)
            if key not in aliases:
                parsed = httpx.URL(canonical)
                convertible = parsed.host in {"xinference", "localhost", "127.0.0.1", "172.17.0.1"} or parsed.host.startswith(("xf-", "vllm-", "sglang-"))
                generated = set()
                if convertible or container_port:
                    if host_ip is None:
                        host_ip = deployment_service._get_host_ip()
                    if convertible:
                        generated.add(canonical_model_endpoint(
                            deployment_service._to_external_endpoint(canonical, host_ip=host_ip)
                        ))
                    if container_port:
                        generated.add(canonical_model_endpoint(f"http://{host_ip}:{container_port}"))
                aliases[key] = generated
            return canonical_endpoint in aliases[key]
        except HTTPException:
            return False

    for stored_endpoint, owner_id, model_uid, status, mode, port, deployment_id, _, started_at, runtime_managed, read_only in rows:
        # A tenant can save a pending shared deployment at any public URL, and
        # a failed launch already has a generated UID. Neither establishes
        # ownership of that external service. Successful starts and admin-only
        # bindings are durable evidence; preserve legacy running rows as well.
        if mode != "container" and not (
            started_at is not None
            or (runtime_managed is False and read_only is True)
            or (status in {"running", "degraded"} and model_uid)
        ):
            continue
        if matches_endpoint(stored_endpoint, port if mode == "container" else None):
            registered = True
            # Replica lifecycle and health, rather than the aggregate parent
            # status, determine which child endpoints can serve this model.
            if deployment_id not in replica_deployments and owner_id == user_id and status == "running" and model_uid:
                owned.add(model_uid)
    for stored_endpoint, deployment_id, status, health in replica_rows:
        if matches_endpoint(stored_endpoint):
            registered = True
            parent = replica_deployments[deployment_id]
            if parent[1] == user_id and status == "running" and health == "HEALTHY" and parent[2]:
                owned.add(parent[2])
    return registered, owned


def authorize_inference_model(
    endpoint: str,
    model_name: Optional[str],
    user_id: Optional[str],
    *,
    provider: str = "",
    current_user: Optional[Dict[str, Any]] = None,
    owned_model_uids: Optional[Set[str]] = None,
    default_endpoint: Optional[str] = None,
) -> Optional[Set[str]]:
    """Require ownership on shared Xinference; leave external APIs unchanged.

    Runtime clients resolve the current role from the database rather than
    trusting a role stored in a task or supplied by an inline request.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    if current_user is not None and current_user.get("is_admin"):
        return None
    default_endpoint = default_endpoint or settings.xinference_endpoint or "http://xinference:9997"
    canonical_endpoint = canonical_model_endpoint(endpoint, provider)
    canonical_default = canonical_model_endpoint(default_endpoint)
    protected = (
        provider.lower() == "xinference"
        or canonical_endpoint == canonical_default
        or bool(owned_model_uids)
    )
    if not protected:
        requested = httpx.URL(canonical_endpoint)
        default = httpx.URL(canonical_default)
        # The configured shared service remains private even before any model
        # has a deployment row. Its generated HOST_IP alias has the same route.
        has_host_alias = default.host in {"xinference", "localhost", "127.0.0.1", "172.17.0.1"} or default.host.startswith(("xf-", "vllm-", "sglang-"))
        if has_host_alias and (requested.scheme, requested.port, requested.path) == (default.scheme, default.port, default.path):
            from ...deployment.deployment_service import deployment_service

            protected = canonical_endpoint == canonical_model_endpoint(
                deployment_service._to_external_endpoint(default_endpoint)
            )
    if not protected:
        protected, registered_owned = registered_shared_models_for_endpoint(
            endpoint, user_id, provider,
        )
        if protected:
            owned_model_uids = registered_owned
    if not protected:
        return None
    if user_id in (None, "", "anonymous"):
        raise HTTPException(status_code=403, detail="Shared model requests require an authenticated owner")
    if current_user is None:
        from ...auth.user_service import user_service

        current_user = user_service.get_user(user_id)
        if not current_user or not current_user.get("is_active"):
            raise HTTPException(status_code=403, detail="Shared model owner is unavailable")
        if current_user.get("is_admin"):
            return None
    if owned_model_uids is None:
        owned_model_uids = owned_model_uids_for_endpoint(endpoint, user_id, provider)
    if not model_name or model_name not in owned_model_uids:
        raise HTTPException(status_code=403, detail="Shared model requests are limited to owned deployments")
    return {model_name}


def authorize_inference_config(
    config: Dict[str, Any], user_id: Optional[str],
    *, current_user: Optional[Dict[str, Any]] = None,
) -> Optional[Set[str]]:
    """Authorize single and pooled inline model configurations."""
    if not get_settings().auth_enabled or (current_user and current_user.get("is_admin")):
        return None
    binding = _authorize_deployment_binding(config, user_id)
    endpoint = config.get("endpoint") or config.get("api_endpoint")
    model = config.get("model") or config.get("model_name")
    authorized = binding
    if endpoint and binding is None:
        authorized = authorize_inference_model(
            endpoint, model, user_id, provider=config.get("provider") or "",
            current_user=current_user,
        )
    for item in config.get("endpoints") or []:
        authorize_inference_model(item.get("url") or "", item.get("model", model),
                                  user_id, current_user=current_user)
    return authorized


def _authorize_deployment_binding(
    config: Dict[str, Any], user_id: Optional[str],
) -> Optional[Set[str]]:
    """Use stored deployment identity, never caller-supplied framework labels."""
    if config.get("source_type") != "local_deployed" or not (
        config.get("deployment_id") or config.get("deployment_replica_id")
    ):
        return None
    from ...deployment.deployment_service import deployment_service

    deployment_id = config.get("deployment_id")
    deployment = deployment_service.get_deployment(deployment_id) if deployment_id else None
    if not user_id or not deployment or deployment.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Model deployment is not owned by this user")
    endpoint = deployment.get("xinference_endpoint") or ""
    replicas = deployment.get("replica_instances") or []
    replica_id = config.get("deployment_replica_id")
    if replicas:
        replica = next((item for item in replicas if item.get("replica_id") == replica_id), None)
        if not replica or replica.get("deployment_id") != deployment_id or not (
            replica.get("status") == "running" and replica.get("health_status") == "HEALTHY"
        ):
            raise HTTPException(status_code=403, detail="Model replica is unavailable")
        endpoint = replica.get("endpoint") or ""
    elif replica_id or deployment.get("status") != "running":
        raise HTTPException(status_code=403, detail="Model deployment is unavailable")
    canonical_endpoint = canonical_model_endpoint(
        config.get("endpoint") or config.get("api_endpoint") or "", config.get("provider") or "",
    )
    endpoint_matches = canonical_endpoint == canonical_model_endpoint(endpoint)
    if not endpoint_matches and not replicas and deployment.get("deploy_mode") == "container" and deployment.get("port"):
        # Legacy/single-container config creation publishes HOST_IP + the
        # registered host port even when the internal endpoint uses another
        # port, scheme, or path. Never take this port from the incoming config.
        endpoint_matches = canonical_endpoint == canonical_model_endpoint(
            f"http://{deployment_service._get_host_ip()}:{deployment['port']}"
        )
    if endpoint and not endpoint_matches:
        endpoint_matches = canonical_endpoint == canonical_model_endpoint(
            deployment_service._to_external_endpoint(endpoint)
        )
    model_name = config.get("model") or config.get("model_name")
    if not model_name or model_name != deployment.get("model_uid") or not endpoint_matches:
        raise HTTPException(status_code=403, detail="Model config does not match its deployment binding")
    return {model_name}
