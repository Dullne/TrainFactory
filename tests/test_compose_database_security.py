import ast
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).parents[1]
COMPOSE_PATH = ROOT_DIR / "docker" / "docker-compose.yml"


def _environment(service):
    environment = service["environment"]
    if isinstance(environment, dict):
        return "\n".join(f"{name}={value}" for name, value in environment.items())
    return "\n".join(environment)


def _has_exact_migration_broad_exception_contract(source: str) -> bool:
    expected_messages = {
        "_controlled_runner_url": "Controlled migration URL is invalid",
        "_verify_controlled_connection": (
            "Controlled migration database identity mismatch"
        ),
    }
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if set(expected_messages) - set(functions):
        return False

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    def catches_broad_exception(handler_type) -> bool:
        if handler_type is None:
            return True
        if isinstance(handler_type, ast.Name):
            return handler_type.id in {"Exception", "BaseException"}
        if isinstance(handler_type, ast.Tuple):
            return any(catches_broad_exception(item) for item in handler_type.elts)
        return False

    broad_handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and catches_broad_exception(node.type)
    ]
    if len(broad_handlers) != len(expected_messages):
        return False

    approved_assign = ast.dump(
        ast.Assign(
            targets=[ast.Name(id="valid", ctx=ast.Store())],
            value=ast.Constant(value=False),
        ),
        include_attributes=False,
    )
    seen = set()
    for handler in broad_handlers:
        owner = parents.get(handler)
        while owner is not None and not isinstance(
            owner, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            owner = parents.get(owner)
        if owner is None or functions.get(owner.name) is not owner:
            return False
        if owner.name not in expected_messages or owner.name in seen:
            return False
        if not isinstance(handler.type, ast.Name) or handler.type.id != "Exception":
            return False
        if len(handler.body) != 1 or (
            ast.dump(handler.body[0], include_attributes=False) != approved_assign
        ):
            return False
        seen.add(owner.name)

    for function_name, expected_message in expected_messages.items():
        fixed_raises = [
            node
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "RuntimeError"
            and len(node.exc.args) == 1
            and isinstance(node.exc.args[0], ast.Constant)
            and node.exc.args[0].value == expected_message
            and isinstance(node.cause, ast.Constant)
            and node.cause.value is None
        ]
        if len(fixed_raises) != 1:
            return False
    return seen == set(expected_messages)


def test_api_uses_required_dedicated_database_credentials():
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))

    for service_name in ("train-factory-api", "train-factory-api-dev"):
        environment = _environment(compose["services"][service_name])
        assert "MYSQL_APP_USER=${MYSQL_APP_USER:-trainfactory_app}" in environment
        assert "MYSQL_APP_PASSWORD=${MYSQL_APP_PASSWORD:-}" in environment
        assert "MYSQL_HOST=mysql" in environment
        assert "MYSQL_DATABASE=train_factory" in environment
        assert not any(
            line.startswith("MYSQL_URL=") for line in environment.splitlines()
        )


def test_mysql_requires_distinct_secrets_and_is_not_host_published():
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    mysql = compose["services"]["mysql"]
    environment = _environment(mysql)
    healthcheck = " ".join(mysql["healthcheck"]["test"])

    assert "ports" not in mysql
    assert "MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-}" in environment
    assert "MYSQL_USER=${MYSQL_APP_USER:-trainfactory_app}" in environment
    assert "MYSQL_PASSWORD=${MYSQL_APP_PASSWORD:-}" in environment
    assert "$${MYSQL_USER}" in healthcheck
    assert "MYSQL_PWD" in healthcheck
    assert "$${MYSQL_PASSWORD_FILE" in healthcheck
    assert "$${MYSQL_PASSWORD" in healthcheck
    assert "$${MYSQL_DATABASE}" in healthcheck
    assert "SELECT 1" in healthcheck
    assert "MYSQL_ROOT_PASSWORD" not in healthcheck
    assert '-p"$${MYSQL_PASSWORD}"' not in healthcheck
    assert "default-authentication-plugin" not in mysql["command"]
    assert "trainfactory123" not in COMPOSE_PATH.read_text(encoding="utf-8")


def test_secret_overlay_removes_all_direct_secret_aliases_and_uses_host_paths():
    source = (ROOT_DIR / "docker" / "docker-compose.secrets.yml").read_text(
        encoding="utf-8"
    )

    for variable in (
        "MYSQL_ROOT_PASSWORD",
        "MYSQL_PASSWORD",
        "MYSQL_URL",
        "MYSQL_APP_PASSWORD",
        "JWT_SECRET_KEY",
        "DEFAULT_ADMIN_PASSWORD",
    ):
        assert f"{variable}: !reset null" in source
    for variable in (
        "MYSQL_ROOT_PASSWORD_SECRET_PATH",
        "MYSQL_APP_PASSWORD_SECRET_PATH",
        "MYSQL_URL_SECRET_PATH",
        "JWT_SECRET_KEY_SECRET_PATH",
        "DEFAULT_ADMIN_PASSWORD_SECRET_PATH",
    ):
        assert f'file: "${{{variable}:?' in source
    assert "file: ./secrets/" not in source


def test_example_environment_has_no_usable_database_password_or_host_port():
    example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "MYSQL_ROOT_PASSWORD=\n" in example
    assert "MYSQL_APP_USER=trainfactory_app\n" in example
    assert "MYSQL_APP_PASSWORD=\n" in example
    assert "MYSQL_PORT=" not in example
    assert "trainfactory123" not in example


def test_example_environment_defaults_to_loopback_http_transport():
    example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "HOST_BIND_ADDRESS=127.0.0.1\n" in example
    assert "PUBLIC_BASE_URL=http://localhost:3000\n" in example
    assert "AUTH_COOKIE_SECURE=false\n" in example
    assert "HOST_BIND_ADDRESS=0.0.0.0" in example
    assert "PUBLIC_BASE_URL=https://train.example.com" in example
    assert "AUTH_COOKIE_SECURE=true" in example


def test_api_image_does_not_embed_database_credentials():
    dockerfile = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV MYSQL_URL=" not in dockerfile
    assert "trainfactory123" not in dockerfile


def test_api_image_caches_dependencies_before_copying_application_source():
    dockerfile = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")

    dependency_install = dockerfile.index("python -m pip --isolated install")
    source_copy = dockerfile.index("COPY train_factory/ train_factory/")

    assert dockerfile.index("COPY requirements/runtime.lock") < dependency_install < source_copy
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "--require-hashes --only-binary=:all: -r requirements/runtime.lock" in dockerfile
    assert "python -m pip --isolated install --no-deps --no-build-isolation ." in dockerfile
    assert " -e ." not in dockerfile


def test_runtime_database_url_defaults_never_use_root():
    settings_source = (ROOT_DIR / "train_factory" / "config" / "settings.py").read_text(
        encoding="utf-8"
    )
    migration_source = (
        ROOT_DIR / "train_factory" / "storage" / "migrations" / "env.py"
    ).read_text(encoding="utf-8")

    assert "mysql+pymysql://root:" not in settings_source
    assert "mysql+pymysql://root:" not in migration_source
    assert 'os.environ.get("MYSQL_URL")' in migration_source
    assert 'os.environ.get("DATABASE_URL")' in migration_source
    assert 'raise RuntimeError("Controlled migration URL is invalid") from None' in (
        migration_source
    )
    assert (
        'raise RuntimeError("Controlled migration database identity mismatch") '
        "from None"
    ) in migration_source
    assert "mysql+pymysql://trainfactory_app@localhost" not in migration_source

    assert _has_exact_migration_broad_exception_contract(migration_source)


def test_runtime_database_broad_exception_contract_rejects_other_scopes():
    migration_source = (
        ROOT_DIR / "train_factory" / "storage" / "migrations" / "env.py"
    ).read_text(encoding="utf-8")
    module_handler = "\ntry:\n    pass\nexcept Exception:\n    pass\n"
    class_handler = (
        "\nclass Injected:\n"
        "    def probe(self):\n"
        "        try:\n"
        "            pass\n"
        "        except Exception:\n"
        "            pass\n"
    )
    bare_handler = "\ntry:\n    pass\nexcept:\n    pass\n"
    base_handler = "\ntry:\n    pass\nexcept BaseException:\n    pass\n"
    tuple_handler = "\ntry:\n    pass\nexcept (Exception, OSError):\n    pass\n"

    for injected_handler in (
        module_handler,
        class_handler,
        bare_handler,
        base_handler,
        tuple_handler,
    ):
        assert not _has_exact_migration_broad_exception_contract(
            migration_source + injected_handler
        )


def test_alembic_uses_the_current_path_separator_configuration_key():
    alembic_config = (
        ROOT_DIR / "train_factory" / "storage" / "migrations" / "alembic.ini"
    ).read_text(encoding="utf-8")

    assert "path_separator = os" in alembic_config.splitlines()


def test_example_requires_url_safe_database_secrets():
    example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "URL-safe hexadecimal" in example


def test_compose_has_no_public_object_store_credentials_or_public_minio_ports():
    source = COMPOSE_PATH.read_text(encoding="utf-8")
    compose = yaml.safe_load(source)

    assert "minioadmin" not in source.lower()
    assert "${MINIO_ACCESS_KEY:?" not in source
    assert "${MINIO_SECRET_KEY:?" not in source

    for service_name in ("train-factory-api", "train-factory-api-dev"):
        environment = _environment(compose["services"][service_name])
        assert "MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-}" in environment
        assert "MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-}" in environment
        assert "API_WORKERS=${API_WORKERS:-1}" in environment

    minio = compose["services"]["minio"]
    assert minio["environment"] == {
        "MINIO_ROOT_USER": "${MINIO_ACCESS_KEY:-}",
        "MINIO_ROOT_PASSWORD": "${MINIO_SECRET_KEY:-}",
    }
    startup = " ".join(minio["entrypoint"])
    assert "MINIO_ROOT_USER is required" in startup
    assert "MINIO_ROOT_PASSWORD is required" in startup
    assert "vector" in minio["profiles"]
    assert minio["ports"] == [
        {
            "target": 9001,
            "published": "${MINIO_CONSOLE_PORT:-9001}",
            "host_ip": "${HOST_BIND_ADDRESS:-127.0.0.1}",
            "protocol": "tcp",
        }
    ]


def test_example_environment_requires_opt_in_object_store_credentials():
    example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "API_WORKERS=1\n" in example
    assert "MINIO_ACCESS_KEY=\n" in example
    assert "MINIO_SECRET_KEY=\n" in example
    assert "minioadmin" not in example.lower()
    assert "required when STORAGE_BACKEND=s3" in example


def test_optional_unauthenticated_service_ports_are_loopback_only():
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))

    assert compose["services"]["xinference"]["ports"] == [
        {
            "target": 9997,
            "published": "${XINFERENCE_PORT:-9997}",
            "host_ip": "${HOST_BIND_ADDRESS:-127.0.0.1}",
            "protocol": "tcp",
        }
    ]
    assert compose["services"]["milvus"]["ports"] == [
        {
            "target": 19530,
            "published": "${MILVUS_PORT:-19530}",
            "host_ip": "${HOST_BIND_ADDRESS:-127.0.0.1}",
            "protocol": "tcp",
        },
        {
            "target": 9091,
            "published": "${MILVUS_GRPC_PORT:-9091}",
            "host_ip": "${HOST_BIND_ADDRESS:-127.0.0.1}",
            "protocol": "tcp",
        },
    ]


def test_api_build_context_excludes_local_tool_and_planning_metadata():
    patterns = {
        line.strip()
        for line in (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {
        ".agents",
        ".claude",
        ".superpowers",
        ".codex",
        ".mem",
        ".mcp.json",
        ".gitignore.local",
        "AGENTS.md",
        "plan.md",
        "docs",
    } <= patterns


def test_shared_ignore_files_exclude_runtime_secret_and_build_artifacts():
    gitignore = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore = set(
        (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8").splitlines()
    )

    assert "/.runtime/" in gitignore
    assert "/secrets/*" in gitignore
    assert "!/secrets/README.md" in gitignore
    assert {
        "/secrets",
        ".runtime",
        ".env*",
        "uv.lock",
        "*.bak",
        "*.backup",
        "*~",
        "*.orig",
        ".coverage*",
        "htmlcov",
        "junit*.xml",
        "web/playwright-report",
        "web/test-results",
    } <= dockerignore
    assert "!.env.example" not in dockerignore


def test_example_documents_all_file_overlay_host_path_variables():
    example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")

    for variable in (
        "MYSQL_ROOT_PASSWORD_SECRET_PATH",
        "MYSQL_APP_PASSWORD_SECRET_PATH",
        "MYSQL_URL_SECRET_PATH",
        "JWT_SECRET_KEY_SECRET_PATH",
        "DEFAULT_ADMIN_PASSWORD_SECRET_PATH",
    ):
        assert f"{variable}=\n" in example


def test_secret_directory_contains_only_safe_generation_instructions():
    readme = (ROOT_DIR / "secrets" / "README.md").read_text(encoding="utf-8")

    assert "materialize_compose_secrets.py create" in readme
    assert ".runtime" in readme
    assert "Do not place secret values" in readme
