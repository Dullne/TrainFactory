import importlib
import fnmatch
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).parents[1]
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PROVIDER_LINES = (
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cudnn-cu12==9.1.0.70",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147",
    "nvidia-cusolver-cu12==11.6.1.9",
    "nvidia-cusparse-cu12==12.3.1.170",
    "nvidia-cusparselt-cu12==0.6.2",
    "nvidia-nccl-cu12==2.21.5",
    "nvidia-nvjitlink-cu12==12.4.127",
    "nvidia-nvtx-cu12==12.4.127",
    "setuptools==75.8.0",
    "torch==2.6.0+cu124",
    "torchaudio==2.6.0+cu124",
    "torchelastic==0.2.2",
    "torchvision==0.21.0+cu124",
    "triton==3.2.0",
    "wheel==0.45.1",
)
PROVIDER_NAMES = tuple(line.split("==", 1)[0] for line in PROVIDER_LINES)


def _module(name):
    return importlib.import_module(f"scripts.{name}")


def test_lock_metadata_pins_complete_linux_resolution_contract():
    metadata = tomllib.loads(
        (REQUIREMENTS_DIR / "lock-metadata.toml").read_text(encoding="utf-8")
    )

    assert metadata["tool"] == {
        "uv_version": "0.11.19",
        "python_version": "3.11",
        "python_platform": "x86_64-unknown-linux-gnu",
        "default_index": "https://pypi.org/simple",
        "cpu_index": "https://download.pytorch.org/whl/cpu",
        "torch_backend": "cpu",
        "index_strategy": "first-index",
        "resolution": "highest",
        "prerelease": "if-necessary",
        "keyring_provider": "disabled",
        "exclude_newer": "2026-08-16T00:00:00Z",
        "only_binary": ":all:",
        "no_sources": True,
        "no_header": True,
    }
    assert metadata["runtime"] == {
        "inputs": ["pyproject.toml"],
        "overrides": "requirements/runtime-overrides.txt",
        "providers": "requirements/base-image-provided.txt",
        "output": "requirements/runtime.lock",
    }
    assert metadata["test_cpu"] == {
        "inputs": ["pyproject.toml", "requirements/test-build-requirements.in"],
        "extra": ["dev"],
        "overrides": "requirements/test-cpu-overrides.txt",
        "output": "requirements/test-cpu.lock",
    }


def test_qwen_evaluation_extra_keeps_used_mteb_without_sdist_only_full_stack():
    project = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT_DIR / "train_factory").rglob("*.py")
    )

    assert '"qwen3-rerank-trainer[eval]>=0.1.0,<0.3.0"' in project
    assert "qwen3-rerank-trainer[full]" not in project
    assert "MTEBRerankEvaluator" in source


def test_provider_snapshot_overrides_and_build_inputs_are_exact_and_lf():
    snapshot_bytes = (REQUIREMENTS_DIR / "base-image-provided.txt").read_bytes()
    snapshot = snapshot_bytes.decode("utf-8").splitlines()
    overrides = (
        (REQUIREMENTS_DIR / "runtime-overrides.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert b"\r" not in snapshot_bytes
    assert snapshot == sorted(PROVIDER_LINES)
    assert [line.split("==", 1)[0] for line in overrides] == sorted(PROVIDER_NAMES)
    assert dict(line.split("==", 1) for line in overrides) == {
        name: version.removesuffix("+cu124")
        for name, version in (line.split("==", 1) for line in PROVIDER_LINES)
    }
    assert (REQUIREMENTS_DIR / "test-build-requirements.in").read_text(
        encoding="utf-8"
    ) == "setuptools==75.8.0\nwheel==0.45.1\n"
    assert (REQUIREMENTS_DIR / "test-cpu-overrides.txt").read_text(
        encoding="utf-8"
    ) == "torch==2.6.0+cpu\n"
    attributes = (ROOT_DIR / ".gitattributes").read_text(encoding="utf-8")
    for pattern in (
        "requirements/*.txt",
        "requirements/*.lock",
        "requirements/*.toml",
        "requirements/*.in",
    ):
        assert f"{pattern} text eol=lf" in attributes


@pytest.mark.parametrize("kind", ("runtime", "test_cpu"))
def test_compile_commands_use_only_fixed_metadata_and_wheels(tmp_path, kind):
    module = _module("check_dependency_locks")
    metadata = module.load_lock_metadata(ROOT_DIR)
    command = module.build_compile_command(
        metadata,
        kind=kind,
        output_file=tmp_path / f"{kind}.lock",
    )

    assert command[:4] == ["uv", "--no-config", "pip", "compile"]
    for flag, value in (
        ("--python-version", "3.11"),
        ("--python-platform", "x86_64-unknown-linux-gnu"),
        ("--default-index", "https://pypi.org/simple"),
        ("--index-strategy", "first-index"),
        ("--resolution", "highest"),
        ("--prerelease", "if-necessary"),
        ("--keyring-provider", "disabled"),
        ("--exclude-newer", "2026-08-16T00:00:00Z"),
        ("--only-binary", ":all:"),
    ):
        assert command[command.index(flag) + 1] == value
    assert {"--no-sources", "--no-header", "--generate-hashes"} <= set(command)
    if kind == "runtime":
        emitted = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "--no-emit-package"
        ]
        assert emitted == sorted(PROVIDER_NAMES)
        assert "--index" not in command
    else:
        assert command[command.index("--extra") + 1] == "dev"
        assert "--index" not in command
        assert command[command.index("--torch-backend") + 1] == "cpu"


def test_lock_environment_is_minimal_case_insensitive_and_uses_temp_cache():
    module = _module("check_dependency_locks")
    source = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "HTTP_PROXY": "http://proxy.invalid",
        "Uv_InDeX_Private": "private-index-canary",
        "PIP_INDEX_URL": "private-index-canary",
        "PYTHONPATH": "private-index-canary",
        "virtual_env": "private-index-canary",
        "CONDA_PREFIX": "private-index-canary",
        "HOME": "private-home-canary",
        "USERPROFILE": "private-home-canary",
        "NETRC": "private-home-canary",
        "UNRELATED_PRIVATE_VALUE": "private-index-canary",
    }

    environment = module.sanitized_subprocess_environment(
        source,
        cache_directory=Path("safe-cache"),
    )

    assert environment["PATH"] == source["PATH"]
    assert environment["HTTP_PROXY"] == source["HTTP_PROXY"]
    assert environment["UV_CACHE_DIR"] == "safe-cache"
    assert environment["HOME"] == str(Path("safe-cache") / "home")
    assert environment["USERPROFILE"] == str(Path("safe-cache") / "home")
    assert "private-index-canary" not in repr(environment)


def test_lock_policy_rejects_unhashed_provider_windows_and_private_inputs():
    module = _module("check_dependency_locks")
    cases = {
        "unhashed": "safe-package==1.0\n",
        "provider": f"{PROVIDER_NAMES[0]}==1.0 \\\n    --hash=sha256:{'a' * 64}\n",
        "windows": "safe-package==1.0 ; sys_platform == 'win32' \\\n    --hash=sha256:"
        + "a" * 64,
        "credential": "safe-package @ https://user:pass@example.invalid/a.whl",
        "editable": "-e file:///private/local/path",
        "absolute": "safe-package @ C:/private/local/package.whl",
    }

    for category, text in cases.items():
        assert module.validate_lock_text(
            text,
            kind="runtime",
            provider_names=set(PROVIDER_NAMES),
        ) == [f"runtime lock has invalid {category} entry"]


@pytest.mark.parametrize(
    ("text", "category"),
    (
        ("--index-url https://example.invalid/simple\n", "option"),
        ("safe-package @ https://example.invalid/package.whl\n", "direct"),
        ("safe-package>=1 \\\n    --hash=sha256:" + "a" * 64, "requirement"),
        ("safe-package==1\n", "unhashed"),
        ("--editable https://example.invalid/repo.git\n", "editable"),
        ("safe-package @ \\\\server\\private\\package.whl\n", "absolute"),
        ("safe-package @ /private/package.whl\n", "absolute"),
        (
            "safe-package==1 ; os_name == 'nt' \\\n    --hash=sha256:" + "a" * 64,
            "windows",
        ),
        ("safe-package==1 \\\n    --hash=sha256:abcd\n", "hash"),
    ),
)
def test_lock_policy_accepts_only_exact_hashed_registry_requirements(text, category):
    module = _module("check_dependency_locks")

    assert module.validate_lock_text(
        text,
        kind="runtime",
        provider_names=set(PROVIDER_NAMES),
    ) == [f"runtime lock has invalid {category} entry"]


@pytest.mark.parametrize(
    "requirement",
    (
        "nvidia-cublas-cu12==12.4.5.8",
        "triton==3.2.0",
        "torch==2.6.0+cu124",
    ),
)
def test_cpu_lock_policy_rejects_cuda_providers(requirement):
    module = _module("check_dependency_locks")
    text = f"{requirement} \\\n    --hash=sha256:{'a' * 64}\n"

    assert module.validate_lock_text(
        text,
        kind="test_cpu",
        provider_names=set(PROVIDER_NAMES),
    ) == ["test_cpu lock has invalid cpu provider entry"]


def test_generated_locks_keep_mteb_and_cpu_contract_without_sdist_stack():
    module = _module("check_dependency_locks")
    runtime = (REQUIREMENTS_DIR / "runtime.lock").read_text(encoding="utf-8")
    test_cpu = (REQUIREMENTS_DIR / "test-cpu.lock").read_text(encoding="utf-8")

    assert "mteb==2.19.3" in runtime
    assert "mteb==2.19.3" in test_cpu
    assert "evalscope==" not in runtime + test_cpu
    assert "jieba==" not in runtime + test_cpu
    assert "torch==2.6.0+cpu" in test_cpu
    assert "nvidia-ml-py==" in test_cpu
    assert not any(
        f"{name}==" in test_cpu
        for name in PROVIDER_NAMES
        if name.startswith("nvidia-") or name == "triton"
    )
    assert "+cu" not in test_cpu
    assert (
        module.validate_lock_text(
            runtime,
            kind="runtime",
            provider_names=set(PROVIDER_NAMES),
        )
        == []
    )
    assert (
        module.validate_lock_text(
            test_cpu,
            kind="test_cpu",
            provider_names=set(PROVIDER_NAMES),
        )
        == []
    )


def test_write_mode_validates_both_locks_before_atomic_replacement(
    tmp_path, monkeypatch
):
    module = _module("check_dependency_locks")
    root = tmp_path / "repo"
    requirements = root / "requirements"
    requirements.mkdir(parents=True)
    runtime = requirements / "runtime.lock"
    test_cpu = requirements / "test-cpu.lock"
    runtime.write_text("old-runtime", encoding="utf-8")
    test_cpu.write_text("old-test", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "generate_lock_pair",
        lambda *args, **kwargs: {
            "runtime": "valid-runtime",
            "test_cpu": "invalid-test",
        },
    )
    monkeypatch.setattr(
        module,
        "validate_lock_text",
        lambda text, **kwargs: []
        if text.startswith("valid")
        else ["fixed invalid lock"],
    )

    with pytest.raises(ValueError, match="dependency lock validation failed"):
        module.update_locks_transactionally(root, metadata=object())

    assert runtime.read_text(encoding="utf-8") == "old-runtime"
    assert test_cpu.read_text(encoding="utf-8") == "old-test"


def test_lock_pair_rollback_restores_original_bytes_after_second_replace_failure(
    tmp_path,
    monkeypatch,
):
    module = _module("check_dependency_locks")
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    runtime = requirements / "runtime.lock"
    test_cpu = requirements / "test-cpu.lock"
    runtime.write_bytes(b"old-runtime\r\n")
    test_cpu.write_bytes(b"old-test\r\n")
    original = module._write_bytes_atomically
    failed = False

    def fail_second(path, content):
        nonlocal failed
        if path == test_cpu and not failed:
            failed = True
            raise OSError("injected second replace failure")
        return original(path, content)

    monkeypatch.setattr(module, "_write_bytes_atomically", fail_second)
    with pytest.raises(ValueError, match="dependency lock update failed"):
        module._replace_lock_pair(
            tmp_path,
            {"runtime": "new-runtime\n", "test_cpu": "new-test\n"},
        )

    assert runtime.read_bytes() == b"old-runtime\r\n"
    assert test_cpu.read_bytes() == b"old-test\r\n"


def test_lock_pair_reports_fixed_error_when_byte_exact_rollback_fails(
    tmp_path,
    monkeypatch,
):
    module = _module("check_dependency_locks")
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    runtime = requirements / "runtime.lock"
    test_cpu = requirements / "test-cpu.lock"
    runtime.write_bytes(b"old-runtime\r\n")
    test_cpu.write_bytes(b"old-test\r\n")
    calls = 0

    def fail_update_then_rollback(path, content):
        nonlocal calls
        calls += 1
        if calls == 1:
            path.write_bytes(content)
            return
        raise OSError("private rollback canary")

    monkeypatch.setattr(module, "_write_bytes_atomically", fail_update_then_rollback)
    with pytest.raises(ValueError, match="^dependency lock rollback failed$") as error:
        module._replace_lock_pair(
            tmp_path,
            {"runtime": "new-runtime\n", "test_cpu": "new-test\n"},
        )

    assert "private rollback canary" not in str(error.value)


def test_lock_generator_subprocesses_have_fixed_timeouts(tmp_path, monkeypatch):
    module = _module("check_dependency_locks")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["uv", "--version"]:
            return subprocess.CompletedProcess(command, 0, "uv 0.11.19\n", "")
        output = Path(command[command.index("--output-file") + 1])
        output.write_text(
            "safe==1 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_provider_names", lambda metadata: [])
    assert module._uv_version({}) == "0.11.19"
    metadata = module.LockMetadata(tmp_path, module._fixed_metadata())
    module.generate_lock(
        tmp_path,
        metadata,
        kind="runtime",
        output_file=tmp_path / "runtime.lock",
        environment={},
    )

    assert calls[0][1]["timeout"] == 30
    assert calls[1][1]["timeout"] == 600


def test_direct_lock_script_ignores_pythonpath_shadow_package(tmp_path):
    shadow = tmp_path / "scripts"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "shadow-executed"
    (shadow / "check_base_image_providers.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise SystemExit(73)\n",
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('sitecustomize')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-I", "scripts/check_dependency_locks.py", "--help"],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert not marker.exists()


def test_lock_checker_cli_never_echoes_unknown_argument(capsys):
    module = _module("check_dependency_locks")
    canary = "private-cli-argument-canary"

    assert module.main(["--unknown", canary]) == 2

    captured = capsys.readouterr()
    assert canary not in captured.out
    assert canary not in captured.err


def test_snapshot_requires_digest_and_uses_container_metadata(tmp_path):
    module = _module("snapshot_base_providers")
    image = "pytorch/pytorch:2.6.0@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="digest"):
        module.snapshot_command("pytorch/pytorch:latest")
    command = module.snapshot_command(image)
    assert command[:5] == ["docker", "run", "--rm", "--entrypoint", "python"]
    assert command[5] == image
    assert "importlib.metadata" in command[-1]

    snapshot = tmp_path / "providers.txt"
    snapshot.write_text(
        "\n".join(sorted(PROVIDER_LINES)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert module.compare_snapshot(
        "\n".join(reversed(PROVIDER_LINES)) + "\n",
        snapshot,
        write=False,
    )


def test_base_provider_checker_requires_exact_versions():
    module = _module("check_base_image_providers")
    expected = "\n".join(PROVIDER_LINES) + "\n"
    installed = dict(line.split("==", 1) for line in PROVIDER_LINES)
    assert module.validate_installed_providers(expected, installed) == []
    installed[PROVIDER_NAMES[0]] = "private-version-canary"
    assert module.validate_installed_providers(expected, installed) == [
        f"provider {PROVIDER_NAMES[0]} version mismatch"
    ]


def test_provider_discovery_rejects_duplicate_canonical_distribution(monkeypatch):
    module = _module("check_base_image_providers")

    class Distribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: (
            Distribution("torch", "2.6.0+cu124"),
            Distribution("Torch", "private-duplicate-canary"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        module.installed_provider_versions()


def test_snapshot_cli_is_private_times_out_and_cleans_failed_write(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module("snapshot_base_providers")
    canary = "private-cli-canary"
    assert module.main(["--unknown", canary]) == 2
    captured = capsys.readouterr()
    assert canary not in captured.out
    assert canary not in captured.err

    with pytest.raises(ValueError, match="digest"):
        module.snapshot_command("-invalid/image@sha256:" + "a" * 64)

    snapshot = tmp_path / "providers.txt"
    snapshot.write_text(
        "\n".join(sorted(PROVIDER_LINES)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    calls = []

    class Completed:
        returncode = 0
        stdout = "\n".join(sorted(PROVIDER_LINES)) + "\n"

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Completed(),
    )
    image = "pytorch/pytorch:2.6.0@sha256:" + "a" * 64
    assert module.main(["--image", image, "--snapshot", str(snapshot)]) == 0
    assert calls[0][1]["timeout"] == 300

    monkeypatch.setattr(
        module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(OSError):
        module.compare_snapshot(
            "\n".join(PROVIDER_LINES) + "\n",
            snapshot,
            write=True,
        )
    assert not (tmp_path / ".providers.txt.tmp").exists()


def test_dependency_dockerfiles_are_binary_only_cpu_gpu_separated_and_identified():
    production = (ROOT_DIR / "docker" / "Dockerfile").read_text(encoding="utf-8")
    test_image = (ROOT_DIR / "docker" / "Dockerfile.test").read_text(encoding="utf-8")

    assert "ARG API_BASE_IMAGE\n" in production
    assert "FROM ${API_BASE_IMAGE} AS runtime" in production
    assert (
        "apt-get" not in production
        and " curl" not in production
        and " git" not in production
    )
    assert (
        "--require-hashes --only-binary=:all: -r requirements/runtime.lock"
        in production
    )
    assert (
        "python -m pip --isolated install --no-deps --no-build-isolation ."
        in production
    )
    assert "rm -rf build train_factory.egg-info" in production
    assert (
        "COPY --chmod=0555 docker/entrypoint.sh /app/docker/entrypoint.sh" in production
    )
    assert "urllib.request" in production
    assert production.count("RUN --mount=type=cache,target=/root/.cache/pip") >= 2

    assert "ARG API_TEST_BASE_IMAGE\n" in test_image
    assert "FROM ${API_TEST_BASE_IMAGE}" in test_image
    assert "pytorch/pytorch" not in test_image
    assert (
        "--require-hashes --only-binary=:all: -r requirements/test-cpu.lock"
        in test_image
    )
    assert "--index-url https://pypi.org/simple" in test_image
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in test_image
    assert "setuptools==75.8.0" in test_image and "wheel==0.45.1" in test_image
    assert "AS test" in test_image and "AS api-smoke" in test_image
    assert test_image.count("RUN --mount=type=cache,target=/root/.cache/pip") >= 2
    assert "rm -rf build train_factory.egg-info" in test_image
    assert test_image.count("ENV GPU_PREFLIGHT_MODE=off") >= 2
    test_base, targets = test_image.split("FROM test-base AS test", 1)
    test_target, api_target = targets.split("FROM test-base AS api-smoke", 1)
    dummy_mysql_url = (
        "ENV MYSQL_URL=mysql+pymysql://trainfactory_app@localhost:3306/train_factory"
    )
    assert dummy_mysql_url not in test_base
    assert dummy_mysql_url in test_target
    assert dummy_mysql_url not in api_target
    for argument in ("BUILD_VERSION", "VCS_REF", "VCS_DATE", "SOURCE_REPOSITORY"):
        assert f"ARG {argument}\n" in production
        assert f"ARG {argument}\n" in test_image
        assert f'test -n "${{{argument}}}"' in production
        assert f'test -n "${{{argument}}}"' in test_image
    assert production.count("org.opencontainers.image.") >= 4
    assert test_image.count("org.opencontainers.image.") >= 8


def test_test_dockerignore_is_sensitive_allowlist_for_copy_sources():
    ignore_lines = (
        (ROOT_DIR / "docker" / "Dockerfile.test.dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    dockerfile = (ROOT_DIR / "docker" / "Dockerfile.test").read_text(encoding="utf-8")
    assert (ROOT_DIR / "scripts" / "verify_inference_upstream.py").is_file()
    assert ignore_lines[0] == "*"
    assert all("docs/" not in line for line in ignore_lines)
    assert "production-web-release" not in dockerfile
    assert {
        ".env*",
        ".runtime",
        "secrets",
        "uv.lock",
        "output",
        "*.bak",
        "*.backup",
        "*~",
        "*.orig",
        ".coverage*",
        "htmlcov",
        "junit*.xml",
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**",
        "**/__pycache__/**",
        "**/*.py[cod]",
        "**/*.log",
        "**/*.egg-info/**",
        "**/build/**",
        "**/dist/**",
    } <= set(ignore_lines)
    for required in (
        "!pyproject.toml",
        "!README.md",
        "!README_CN.md",
        "!.env.example",
        "!.dockerignore",
        "!.gitignore",
        "!.gitattributes",
        "!requirements/",
        "!requirements/test-cpu.lock",
        "!requirements/base-image-provided.txt",
        "!requirements/lock-metadata.toml",
        "!requirements/runtime-overrides.txt",
        "!requirements/runtime.lock",
        "!requirements/test-build-requirements.in",
        "!requirements/test-cpu-overrides.txt",
        "!train_factory/",
        "!train_factory/**",
        "!tests/",
        "!tests/**",
        "!.github/",
        "!.github/workflows/",
        "!.github/workflows/ci.yml",
        "!scripts/",
        "!scripts/check_base_image_providers.py",
        "!scripts/check_dependency_locks.py",
        "!scripts/snapshot_base_providers.py",
        "!scripts/materialize_compose_secrets.py",
        "!scripts/scan_tracked_secrets.py",
        "!scripts/validate_compose_config.py",
        "!scripts/build_release.py",
        "!scripts/run_mysql_migration_tests.py",
        "!scripts/compose_manifest.py",
        "!scripts/compose_release.py",
        "!scripts/promote_web_release.py",
        "!scripts/verify_deployment.py",
        "!scripts/check_pytest_partitions.py",
        "!scripts/validate_test_report.py",
        "!scripts/run_compose_smoke.py",
        "!scripts/verify_inference_upstream.py",
        "!docker/",
        "!docker/entrypoint.sh",
        "!docker/Dockerfile",
        "!docker/Dockerfile.test",
        "!docker/Dockerfile.test.dockerignore",
        "!docker/docker-compose.yml",
        "!docker/docker-compose.secrets.yml",
        "!docker/docker-compose.cpu.yml",
        "!docker/docker-compose.gpu-compat.yml",
        "!docker/docker-compose.deployment.yml",
        "!docker/docker-compose.release.yml",
        "!docker/docker-compose.verify.yml",
        "!docker/images.lock.env",
        "!docker/init.sql",
        "!docker/inference-contracts/",
        "!docker/inference-contracts/**",
        "!docker/sglang-templates/",
        "!docker/sglang-templates/**",
        "!docker/xinference-patches/",
        "!docker/xinference-patches/**",
        "!web/",
        "!web/nginx.conf",
        "!web/vite.config.ts",
        "!web/Dockerfile",
        "!secrets/",
        "!secrets/README.md",
    ):
        assert required in ignore_lines
    for source in (
        "pyproject.toml",
        "README.md",
        "README_CN.md",
        ".env.example",
        ".dockerignore",
        ".gitignore",
        ".gitattributes",
        "requirements/test-cpu.lock",
        "requirements/",
        "train_factory/",
        "tests/",
        ".github/workflows/ci.yml",
        "scripts/check_base_image_providers.py",
        "scripts/check_dependency_locks.py",
        "scripts/snapshot_base_providers.py",
        "scripts/materialize_compose_secrets.py",
        "scripts/scan_tracked_secrets.py",
        "scripts/validate_compose_config.py",
        "scripts/build_release.py",
        "scripts/run_mysql_migration_tests.py",
        "scripts/compose_manifest.py",
        "scripts/compose_release.py",
        "scripts/promote_web_release.py",
        "scripts/verify_deployment.py",
        "scripts/check_pytest_partitions.py",
        "scripts/validate_test_report.py",
        "scripts/run_compose_smoke.py",
        "scripts/verify_inference_upstream.py",
        "docker/entrypoint.sh",
        "docker/Dockerfile",
        "docker/Dockerfile.test",
        "docker/Dockerfile.test.dockerignore",
        "docker/docker-compose.yml",
        "docker/docker-compose.secrets.yml",
        "docker/docker-compose.cpu.yml",
        "docker/docker-compose.gpu-compat.yml",
        "docker/docker-compose.deployment.yml",
        "docker/docker-compose.release.yml",
        "docker/docker-compose.verify.yml",
        "docker/images.lock.env",
        "docker/init.sql",
        "docker/inference-contracts/",
        "docker/sglang-templates/",
        "docker/xinference-patches/",
        "web/nginx.conf",
        "web/vite.config.ts",
        "web/Dockerfile",
        "secrets/README.md",
    ):
        assert source in dockerfile
    assert "COPY scripts/verify_inference_upstream.py scripts/" in dockerfile

    def included(path):
        ignored = False
        normalized = path.rstrip("/")
        for raw_pattern in ignore_lines:
            if not raw_pattern or raw_pattern.startswith("#"):
                continue
            negated = raw_pattern.startswith("!")
            pattern = raw_pattern[1:] if negated else raw_pattern
            pattern = pattern.rstrip("/")
            basename = normalized.rsplit("/", 1)[-1]
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(
                basename, pattern
            ):
                ignored = not negated
        return not ignored

    for source in (
        "pyproject.toml",
        "README.md",
        "README_CN.md",
        ".env.example",
        ".dockerignore",
        ".gitignore",
        ".gitattributes",
        "requirements/test-cpu.lock",
        "requirements/runtime.lock",
        "requirements/lock-metadata.toml",
        "train_factory/api/server.py",
        "tests/test_dependency_locks.py",
        ".github/workflows/ci.yml",
        "scripts/check_base_image_providers.py",
        "scripts/check_dependency_locks.py",
        "scripts/snapshot_base_providers.py",
        "scripts/materialize_compose_secrets.py",
        "scripts/scan_tracked_secrets.py",
        "scripts/validate_compose_config.py",
        "scripts/build_release.py",
        "scripts/run_mysql_migration_tests.py",
        "scripts/compose_manifest.py",
        "scripts/compose_release.py",
        "scripts/promote_web_release.py",
        "scripts/verify_deployment.py",
        "scripts/check_pytest_partitions.py",
        "scripts/validate_test_report.py",
        "scripts/run_compose_smoke.py",
        "scripts/verify_inference_upstream.py",
        "docker/entrypoint.sh",
        "docker/Dockerfile",
        "docker/Dockerfile.test",
        "docker/Dockerfile.test.dockerignore",
        "docker/docker-compose.yml",
        "docker/docker-compose.secrets.yml",
        "docker/docker-compose.cpu.yml",
        "docker/docker-compose.gpu-compat.yml",
        "docker/docker-compose.deployment.yml",
        "docker/docker-compose.release.yml",
        "docker/docker-compose.verify.yml",
        "docker/images.lock.env",
        "docker/init.sql",
        "docker/inference-contracts/qwen3-compatibility.json",
        "docker/sglang-templates/qwen3_reranker_no_think.jinja",
        "docker/xinference-patches/sentence_transformers_core.py",
        "docker/xinference-patches/rerank_sentence_transformers_core.py",
        "web/nginx.conf",
        "web/vite.config.ts",
        "web/Dockerfile",
        "secrets/README.md",
    ):
        assert included(source), source

    assert "!README_EN.md" not in ignore_lines
    assert "README_EN.md" not in dockerfile
    assert not any(line.startswith("!.claude") for line in ignore_lines)
    assert ".claude/" not in dockerfile

    assert not included("tests/sync_test_report.md")
    assert not included(".github/workflows/ci-smoke.yml")
    assert ignore_lines.count("tests/sync_test_report.md") == 1
    last_negation = max(
        index for index, pattern in enumerate(ignore_lines) if pattern.startswith("!")
    )
    for artifact_pattern in (
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**",
        "**/__pycache__/**",
        "**/*.py[cod]",
        "**/*.log",
        "**/*.egg-info/**",
        "**/build/**",
        "**/dist/**",
    ):
        assert ignore_lines.index(artifact_pattern) > last_negation
    for sensitive in (
        ".env",
        ".runtime/run/env.list",
        "train_factory/__pycache__/server.cpython-311.pyc",
        "tests/.pytest_cache/v/cache/nodeids",
        "tests/.mypy_cache/cache.json",
        "tests/.ruff_cache/cache",
        "train_factory/runtime.log",
        "train_factory/train_factory.egg-info/PKG-INFO",
        "train_factory/build/wheel.whl",
        "train_factory/dist/package.whl",
        "secrets/mysql_password",
        "uv.lock",
        "output/report.txt",
        "private.backup",
        "htmlcov/index.html",
        "junit-results.xml",
    ):
        assert not included(sensitive), sensitive
