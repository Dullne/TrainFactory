from __future__ import annotations

import asyncio
import hashlib
import json
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, StrictUndefined

from train_factory.deployment import vllm_client as vllm_client_module
from train_factory.deployment.launch_config import xinference_model_launch_overrides


ROOT_DIR = Path(__file__).parents[1]
QWEN3_IMAGE_CONTRACT = (
    ROOT_DIR / "docker" / "inference-contracts" / "qwen3-compatibility.json"
)
XINFERENCE_PATCH_ROOT = ROOT_DIR / "docker" / "xinference-patches"
deployment_service_module = importlib.import_module(
    "train_factory.deployment.deployment_service"
)
model_config_routes_module = importlib.import_module(
    "train_factory.api.routes.model_config_routes"
)


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"results": []}


def test_pinned_inference_images_have_digest_bound_qwen3_contract() -> None:
    assert QWEN3_IMAGE_CONTRACT.exists()
    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))

    assert contract == {
        "schema_version": 2,
        "xinference": {
            "image": (
                "xprobe/xinference:v3.1.0@sha256:"
                "ec41459d15cc1c18842370c267e9c9a12a0001245dea9fe3b939e4075dc18178"
            ),
            "source_layer": (
                "sha256:6d47bd78842f499ad0595774eb2f59deaf0cb5e549a7b6650f20c0530d61159a"
            ),
            "install_layer": (
                "sha256:77185dfe665db5015827d4a56a7f1b45d2c2baf5eec12955930348950d647117"
            ),
            "installed_version": "3.1.1.dev0+g53c2b5e1c.d20260731",
            "installed_metadata_sha256": (
                "a78aa6e2a700ede7ee4fab2279863333b0c2f3e59e75d3224ac43bb69f487b19"
            ),
            "repository": {
                "url": "https://github.com/xorbitsai/inference.git",
                "tag": "v3.1.0",
                "tag_kind": "lightweight",
                "commit": "53c2b5e1c157aa369c20fca7bbf1fb3e9a2e9879",
            },
            "sources": {
                "embedding": {
                    "upstream_path": (
                        "xinference/model/embedding/sentence_transformers/core.py"
                    ),
                    "runtime_path": "model/embedding/sentence_transformers/core.py",
                    "vendored_path": (
                        "docker/xinference-patches/sentence_transformers_core.py"
                    ),
                    "git_blob_sha1": ("0fae7ffc445c231ff5396ac8d2adbc4b24c53958"),
                    "size": 34372,
                    "sha256": (
                        "f9af88270948e5cd2ad108d08fa48215357345c2e3f3b65bcbbf5d0df7bd0b06"
                    ),
                    "patched_sha256": (
                        "034b3f1fa28ab4037c837601f8d9905874e941edc5920944960e9f827eab8921"
                    ),
                },
                "rerank": {
                    "upstream_path": (
                        "xinference/model/rerank/sentence_transformers/core.py"
                    ),
                    "runtime_path": "model/rerank/sentence_transformers/core.py",
                    "vendored_path": (
                        "docker/xinference-patches/rerank_sentence_transformers_core.py"
                    ),
                    "git_blob_sha1": ("4e3b6dbd06fecc4c7cfcd721a900b62a819e690a"),
                    "size": 31894,
                    "sha256": (
                        "ded6772122985699d3c4cd37172f67b80d2a4f138f244c415595dcd7ed6b48cb"
                    ),
                    "patched_sha256": (
                        "bca763d531409fec62e9e4ede6515c16ace7e4c7560cdcff4082f43e3e17dac8"
                    ),
                },
            },
        },
        "sglang": {
            "image": (
                "lmsysorg/sglang:v0.5.17@sha256:"
                "16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155"
            ),
            "linux_amd64_manifest": (
                "sha256:3be8803490a8b899a44f7ab2e22d8f6a1fb877cab52faeb400769a1555317db4"
            ),
            "source_layer": (
                "sha256:a3240c4e46ae106032204eb007dc2d939afcfee6207fe4abd76245ac248127d5"
            ),
            "repository": {
                "url": "https://github.com/sgl-project/sglang.git",
                "tag": "v0.5.17",
                "tag_kind": "annotated",
                "commit": "29481685462732237d80d86076d6563e1f658102",
            },
            "sources": {
                "rerank_template": {
                    "upstream_path": "examples/chat_template/qwen3_reranker.jinja",
                    "git_blob_sha1": ("5ab809eea6a7640eea135c05d01468eb1f931427"),
                    "size": 427,
                    "sha256": (
                        "c6158004ca0cd90048b78eaa325ab1b7909f469dde556f9f378c79ce81008104"
                    ),
                },
                "rerank_handler": {
                    "upstream_path": (
                        "python/sglang/srt/entrypoints/openai/serving_rerank.py"
                    ),
                    "git_blob_sha1": ("7bf410a07ed7f15264e01fd7f1054fa2e7ce7288"),
                    "size": 23427,
                    "sha256": (
                        "5242cd42c8ccba74fbdf13e2c7eee63b5551e3d6bef09933aba213d690cb3329"
                    ),
                },
            },
        },
    }

    images_lock = {}
    for raw_line in (
        (ROOT_DIR / "docker" / "images.lock.env")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            images_lock[key] = value

    assert images_lock["XINFERENCE_IMAGE"] == contract["xinference"]["image"]
    assert images_lock["SGLANG_IMAGE"] == contract["sglang"]["image"]


def test_vllm_rerank_sends_instruction_with_raw_cohere_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def request(method: str, url: str, _user_id=None, **kwargs):
        calls.append((method, url, kwargs))
        return _Response()

    monkeypatch.setattr(vllm_client_module, "request_user_outbound", request)
    client = vllm_client_module.VLLMClient("https://rerank.example.test")

    client.rerank(
        "raw query",
        ["raw document"],
        model="reranker",
        top_n=1,
        instruction="server-side instruction",
    )

    assert calls[0][2]["json"] == {
        "query": "raw query",
        "documents": ["raw document"],
        "model": "reranker",
        "top_n": 1,
        "instruction": "server-side instruction",
    }


def test_vllm_rerank_preserves_explicit_client_timeout_at_request_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    client = vllm_client_module.VLLMClient(
        "https://rerank.example.test",
        timeout=47,
    )

    def request(method: str, path: str, **kwargs):
        captured.update(method=method, path=path, **kwargs)
        return _Response()

    monkeypatch.setattr(client, "_request", request)
    client.rerank("query", ["document"])

    assert captured["timeout"] == 47


@pytest.mark.parametrize("instruction", [None, "", "   "])
def test_vllm_rerank_omits_blank_instruction(
    monkeypatch: pytest.MonkeyPatch,
    instruction: str | None,
) -> None:
    calls: list[dict[str, object]] = []

    def request(_method: str, _url: str, _user_id=None, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(vllm_client_module, "request_user_outbound", request)

    vllm_client_module.VLLMClient("https://rerank.example.test").rerank(
        "raw query",
        ["raw document"],
        instruction=instruction,
    )

    assert "instruction" not in calls[0]["json"]


def test_vllm_rerank_does_not_leak_instruction_in_failure_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_instruction = "instruction-secret-do-not-log"

    def request(*_args, **_kwargs):
        raise RuntimeError(f"transport echoed {secret_instruction}")

    monkeypatch.setattr(vllm_client_module, "request_user_outbound", request)
    client = vllm_client_module.VLLMClient("https://rerank.example.test")

    with caplog.at_level("ERROR", logger=vllm_client_module.__name__):
        with pytest.raises(RuntimeError) as exc_info:
            client.rerank(
                "raw query",
                ["raw document"],
                instruction=secret_instruction,
            )

    assert secret_instruction not in caplog.text
    assert secret_instruction not in str(exc_info.value)
    assert "RuntimeError" in caplog.text


def _load_xinference_compatibility_patcher():
    patcher_path = XINFERENCE_PATCH_ROOT / "apply_qwen3_compatibility.py"
    spec = importlib.util.spec_from_file_location(
        "xinference_compat_patcher", patcher_path
    )
    assert spec is not None and spec.loader is not None
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    return patcher


def _reconstruct_xinference_upstream_sources(
    contract: dict[str, object],
) -> dict[Path, bytes]:
    patcher = _load_xinference_compatibility_patcher()
    sources = {}
    xinference_contract = contract["xinference"]
    assert isinstance(xinference_contract, dict)
    source_contracts = xinference_contract["sources"]
    assert isinstance(source_contracts, dict)
    for relative_path, source_key in patcher.TARGET_PAYLOADS.items():
        source_contract = source_contracts[source_key]
        assert isinstance(source_contract, dict)
        assert Path(source_contract["runtime_path"]) == relative_path
        snapshot = ROOT_DIR / source_contract["vendored_path"]
        source = patcher._canonical_source(snapshot.read_bytes())
        assert hashlib.sha256(source).hexdigest() == source_contract["sha256"]
        sources[relative_path] = source
    return sources


def test_xinference_31_snapshots_and_transforms_match_exact_image_contract() -> None:
    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))
    patcher = _load_xinference_compatibility_patcher()
    assert patcher.TARGET_PAYLOADS == {
        Path("model/embedding/sentence_transformers/core.py"): "embedding",
        Path("model/rerank/sentence_transformers/core.py"): "rerank",
    }
    upstream_sources = _reconstruct_xinference_upstream_sources(contract)

    patched_sources = {}
    for relative_path, source in upstream_sources.items():
        patched = patcher._apply_transformations(relative_path, source)
        patched_sources[relative_path] = patched
        source_key = patcher.TARGET_PAYLOADS[relative_path]
        source_contract = contract["xinference"]["sources"][source_key]
        assert hashlib.sha256(patched).hexdigest() == source_contract["patched_sha256"]
        compile(patched, relative_path.as_posix(), "exec")

    embedding = patched_sources[Path("model/embedding/sentence_transformers/core.py")]
    rerank = patched_sources[Path("model/rerank/sentence_transformers/core.py")]
    assert b"module.pooling_mode_lasttoken = True" in embedding
    assert b"trust_remote_code=ALLOW_MODEL_REMOTE_CODE" in embedding
    assert b"torch.bfloat16" in rerank
    assert b'if "qwen3" in self.model_family.model_name.lower()' in rerank
    assert b"else torch.float16" in rerank
    assert b"true_probabilities = true_probabilities.float()" in rerank
    assert b"_sanitize_remote_code_kwargs(self._kwargs)" in rerank
    assert b"allow_trust_remote_code(self.model_family)" not in embedding
    assert b"allow_trust_remote_code(self.model_family)" not in rerank


def test_public_inference_volume_defaults_use_portable_checkout_root() -> None:
    volumes = {
        "XINFERENCE_PATCH_VOLUME": (
            "/workspace/train-factory/docker/xinference-patches:"
            "/opt/trainfactory/xinference-patches:ro"
        ),
        "XINFERENCE_CONTRACT_VOLUME": (
            "/workspace/train-factory/docker/inference-contracts:"
            "/opt/trainfactory/inference-contracts:ro"
        ),
        "SGLANG_TEMPLATE_VOLUME": (
            "/workspace/train-factory/docker/sglang-templates:"
            "/opt/trainfactory/sglang-templates:ro"
        ),
    }
    compose = (ROOT_DIR / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    release = (ROOT_DIR / "scripts" / "compose_release.py").read_text(
        encoding="utf-8"
    )
    deployer = (
        ROOT_DIR / "train_factory" / "deployment" / "docker_deployer.py"
    ).read_text(encoding="utf-8")

    for variable, volume in volumes.items():
        assert f"{variable}: ${{{variable}:-{volume}}}" in compose
        assert f"{variable}={volume}" in release
        host_path, container_path = volume.split(":", 1)
        assert f'"{host_path}:"' in deployer
        assert f'"{container_path}"' in deployer


def test_public_gitee_fixtures_use_example_owner() -> None:
    source = (ROOT_DIR / "tests" / "test_build_release.py").read_text(
        encoding="utf-8"
    )

    assert source.count("git@gitee.com:example/train-factory.git") == 2
    assert source.count("https://gitee.com/example/train-factory") == 2
    assert source.count("gitee.com") == 4


def test_xinference_31_compose_applies_guarded_compatibility_payloads() -> None:
    compose = (ROOT_DIR / "docker" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "./xinference-patches:/opt/trainfactory/xinference-patches:ro"
    ) in compose
    assert (
        "./inference-contracts:/opt/trainfactory/inference-contracts:ro"
    ) in compose
    assert (
        'entrypoint: ["python", '
        '"/opt/trainfactory/xinference-patches/apply_qwen3_compatibility.py", '
        '"--image-ref", "${XINFERENCE_IMAGE:-xprobe/xinference:v3.1.0@sha256:'
        'ec41459d15cc1c18842370c267e9c9a12a0001245dea9fe3b939e4075dc18178}", '
        '"--contract", '
        '"/opt/trainfactory/inference-contracts/qwen3-compatibility.json", "--"]'
    ) in compose
    assert (
        "XINFERENCE_PATCH_VOLUME: ${XINFERENCE_PATCH_VOLUME:-"
        "/workspace/train-factory/docker/xinference-patches:"
        "/opt/trainfactory/xinference-patches:ro}"
    ) in compose
    assert (
        "XINFERENCE_CONTRACT_VOLUME: ${XINFERENCE_CONTRACT_VOLUME:-"
        "/workspace/train-factory/docker/inference-contracts:"
        "/opt/trainfactory/inference-contracts:ro}"
    ) in compose
    assert (
        "SGLANG_TEMPLATE_VOLUME: ${SGLANG_TEMPLATE_VOLUME:-"
        "/workspace/train-factory/docker/sglang-templates:"
        "/opt/trainfactory/sglang-templates:ro}"
    ) in compose
    example_env = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    for variable in (
        "XINFERENCE_PATCH_VOLUME=",
        "XINFERENCE_CONTRACT_VOLUME=",
        "SGLANG_TEMPLATE_VOLUME=",
    ):
        assert variable in example_env


def test_dynamic_xinference_applies_guarded_compatibility_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from train_factory.deployment.docker_deployer import DockerDeployer

    patch_volume = (
        "/host/xinference-patches:/opt/trainfactory/xinference-patches:ro"
    )
    contract_volume = (
        "/host/inference-contracts:/opt/trainfactory/inference-contracts:ro"
    )
    deployer = DockerDeployer(
        xinference_patch_volume=patch_volume,
        xinference_contract_volume=contract_volume,
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )

    deployer.create_xinference_container(
        container_name="qwen3-embedding",
        port=10001,
        gpu_id=0,
        model_name="Qwen3-Embedding-0.6B",
        model_uid="qwen3-embedding",
        model_path="/models/qwen3-embedding",
        model_type="embedding",
    )

    command = captured["command"]
    assert command[command.index(patch_volume) - 1] == "-v"
    assert command[command.index(contract_volume) - 1] == "-v"
    assert command[command.index("python") - 1] == "--entrypoint"
    image_index = command.index(deployer.xinference_image)
    assert command[image_index + 1 : image_index + 8] == [
        "/opt/trainfactory/xinference-patches/apply_qwen3_compatibility.py",
        "--image-ref",
        deployer.xinference_image,
        "--contract",
        "/opt/trainfactory/inference-contracts/qwen3-compatibility.json",
        "--",
        "bash",
    ]


def test_xinference_runtime_patch_rejects_unknown_preimage(tmp_path: Path) -> None:
    patcher_path = XINFERENCE_PATCH_ROOT / "apply_qwen3_compatibility.py"
    spec = importlib.util.spec_from_file_location("xinference_compat_patcher", patcher_path)
    assert spec is not None and spec.loader is not None
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)

    package_root = tmp_path / "xinference"
    for relative_path in patcher.TARGET_PAYLOADS:
        target = package_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("unknown source", encoding="utf-8")

    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="preimage hash mismatch"):
        patcher.apply_compatibility_payloads(
            package_root=package_root,
            payload_root=XINFERENCE_PATCH_ROOT,
            contract=contract,
        )


def test_xinference_runtime_patch_rejects_contract_runtime_path_mismatch(
    tmp_path: Path,
) -> None:
    patcher = _load_xinference_compatibility_patcher()
    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))
    package_root = tmp_path / "xinference"
    upstream_sources = _reconstruct_xinference_upstream_sources(contract)
    for relative_path, source in upstream_sources.items():
        target = package_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)

    contract["xinference"]["sources"]["embedding"]["runtime_path"] = (
        "model/embedding/sentence_transformers/alternate.py"
    )

    with pytest.raises(RuntimeError, match="runtime path mismatch: embedding"):
        patcher.apply_compatibility_payloads(
            package_root=package_root,
            payload_root=XINFERENCE_PATCH_ROOT,
            contract=contract,
        )


def test_xinference_runtime_patch_rejects_mismatched_image_ref() -> None:
    patcher = _load_xinference_compatibility_patcher()
    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="image reference mismatch"):
        patcher._verify_configured_image(
            contract,
            "xprobe/xinference:v3.1.0@sha256:" + "0" * 64,
        )


def test_xinference_runtime_patch_rejects_schema_v1(tmp_path: Path) -> None:
    patcher = _load_xinference_compatibility_patcher()
    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))
    contract["schema_version"] = 1
    contract_path = tmp_path / "schema-v1.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="unsupported inference compatibility contract",
    ):
        patcher._load_contract(contract_path)


def test_xinference_runtime_patch_verifies_installed_metadata_without_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patcher = _load_xinference_compatibility_patcher()
    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))
    metadata = "Metadata-Version: 2.4\nName: xinference\nVersion: guarded-test\n"
    contract["xinference"]["installed_version"] = "guarded-test"
    contract["xinference"]["installed_metadata_sha256"] = hashlib.sha256(
        metadata.encode("utf-8")
    ).hexdigest()

    class _Distribution:
        version = contract["xinference"]["installed_version"]
        files = None

        @staticmethod
        def read_text(name: str) -> str | None:
            return metadata if name == "METADATA" else None

    monkeypatch.setattr(
        patcher.importlib.metadata,
        "distribution",
        lambda _name: _Distribution(),
    )

    patcher._verify_installed_distribution(contract)


def test_xinference_runtime_patch_is_idempotent_for_declared_sources(
    tmp_path: Path,
) -> None:
    patcher_path = XINFERENCE_PATCH_ROOT / "apply_qwen3_compatibility.py"
    spec = importlib.util.spec_from_file_location(
        "xinference_compat_patcher", patcher_path
    )
    assert spec is not None and spec.loader is not None
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)

    contract = json.loads(QWEN3_IMAGE_CONTRACT.read_text(encoding="utf-8"))
    package_root = tmp_path / "xinference"
    upstream_sources = _reconstruct_xinference_upstream_sources(contract)
    for relative_path, source in upstream_sources.items():
        target = package_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)

    patcher.apply_compatibility_payloads(
        package_root=package_root,
        payload_root=XINFERENCE_PATCH_ROOT,
        contract=contract,
    )
    first_hashes = {
        relative_path: hashlib.sha256(
            (package_root / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in patcher.TARGET_PAYLOADS
    }
    patcher.apply_compatibility_payloads(
        package_root=package_root,
        payload_root=XINFERENCE_PATCH_ROOT,
        contract=contract,
    )
    second_hashes = {
        relative_path: hashlib.sha256(
            (package_root / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in patcher.TARGET_PAYLOADS
    }

    assert first_hashes == second_hashes
    assert first_hashes == {
        relative_path: contract["xinference"]["sources"][source_key]["patched_sha256"]
        for relative_path, source_key in patcher.TARGET_PAYLOADS.items()
    }


def test_sglang_qwen3_reranker_selects_trainfactory_no_think_template() -> None:
    select_template = deployment_service_module._sglang_qwen3_chat_template
    expected = (
        "/opt/trainfactory/sglang-templates/"
        "qwen3_reranker_no_think.jinja"
    )

    assert select_template("reranker", "Qwen/Qwen3-Reranker-4B") == expected
    assert select_template("decoder_reranker", "Qwen/Qwen3-Reranker-4B") == expected
    assert select_template("reranker", "BAAI/bge-reranker-v2-m3") is None
    assert select_template("embedding", "Qwen/Qwen3-Embedding-0.6B") is None


def test_dynamic_sglang_mounts_trainfactory_templates_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from train_factory.deployment.docker_deployer import DockerDeployer

    template_volume = (
        "/host/sglang-templates:/opt/trainfactory/sglang-templates:ro"
    )
    deployer = DockerDeployer(sglang_template_volume=template_volume)
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )

    deployer.create_sglang_container(
        container_name="qwen3-reranker",
        port=10001,
        gpu_ids=(0,),
        server_argv=(
            "sglang",
            "serve",
            "--model-path",
            "/app/models/qwen3-reranker",
            "--chat-template",
            "/opt/trainfactory/sglang-templates/qwen3_reranker_no_think.jinja",
        ),
    )

    command = captured["command"]
    assert command[command.index(template_volume) - 1] == "-v"


def test_sglang_trainfactory_template_renders_exact_no_think_prompt() -> None:
    template_path = (
        ROOT_DIR
        / "docker"
        / "sglang-templates"
        / "qwen3_reranker_no_think.jinja"
    )
    source = template_path.read_text(encoding="utf-8")
    rendered = Environment(undefined=StrictUndefined).from_string(source).render(
        messages=[{"content": "query"}, {"content": "document"}]
    )

    assert rendered == (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and "
        'the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<Instruct>: Given a web search query, retrieve relevant passages that answer "
        "the query\n"
        "<Query>: query\n"
        "<Document>: document<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def test_xinference_qwen3_reranker_launch_forces_bfloat16(monkeypatch) -> None:
    from train_factory.deployment.docker_deployer import DockerDeployer

    deployer = DockerDeployer()
    captured = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )

    deployer.create_xinference_container(
        container_name="qwen3-reranker",
        port=10001,
        model_name="Qwen3-Reranker-4B",
        model_uid="qwen3-reranker",
        model_path="/models/qwen3-reranker",
        model_type="reranker",
    )

    inner_command = captured["command"][-1]
    assert "--torch_dtype bfloat16" in inner_command


def test_xinference_decoder_reranker_uses_rerank_cli_and_bfloat16(
    monkeypatch,
) -> None:
    from train_factory.deployment.docker_deployer import DockerDeployer

    deployer = DockerDeployer()
    captured = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )

    deployer.create_xinference_container(
        container_name="qwen3-decoder-reranker",
        port=10001,
        model_name="Qwen3-Reranker-4B",
        model_uid="qwen3-decoder-reranker",
        model_path="/models/qwen3-reranker",
        model_type="decoder_reranker",
    )

    inner_command = captured["command"][-1]
    assert "--pull=missing" in captured["command"]
    assert "--model-type rerank" in inner_command
    assert "--torch_dtype bfloat16" in inner_command


def test_xinference_generic_reranker_does_not_force_qwen_dtype(monkeypatch) -> None:
    from train_factory.deployment.docker_deployer import DockerDeployer

    deployer = DockerDeployer()
    captured = {}
    monkeypatch.setattr(deployer, "remove_container", lambda _name: False)
    monkeypatch.setattr(
        deployer,
        "_run_container_create",
        lambda command, _name, **_identity: (
            captured.setdefault("command", command) is command,
            "ok",
        ),
    )

    deployer.create_xinference_container(
        container_name="generic-reranker",
        port=10002,
        model_name="bge-reranker-v2-m3",
        model_uid="generic-reranker",
        model_path="/models/generic-reranker",
        model_type="reranker",
    )

    inner_command = captured["command"][-1]
    assert "--torch_dtype" not in inner_command


def test_xinference_rest_launch_uses_qwen3_only_dtype_override() -> None:
    assert xinference_model_launch_overrides(
        model_type="reranker",
        model_family="Qwen3-Reranker-4B",
    ) == {"torch_dtype": "bfloat16"}
    assert xinference_model_launch_overrides(
        model_type="reranker",
        model_family="BAAI/bge-reranker-v2-m3",
    ) == {}
    assert xinference_model_launch_overrides(
        model_type="embedding",
        model_family="Qwen3-Embedding-0.6B",
    ) == {}


def test_xinference_rest_launch_overrides_stale_qwen3_reranker_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def launch_model(self, **kwargs) -> None:
            captured.update(kwargs)

    service = deployment_service_module.DeploymentService()
    monkeypatch.setattr(service, "_get_xinference_client", lambda *_args, **_kwargs: _Client())
    monkeypatch.setattr(
        deployment_service_module,
        "_trusted_model_launch_kwargs",
        lambda _config: {"torch_dtype": "float16", "caller_option": "kept"},
    )
    deployment = SimpleNamespace(
        xinference_endpoint="http://xinference:9997",
        user_id="user-1",
        config={"torch_dtype": "float16"},
        gpu_id=None,
        replica=1,
        gpu_memory_utilization=0.9,
    )

    service._start_external_deployment(
        deployment,
        {
            "model_name": "Qwen/Qwen3-Reranker-4B",
            "model_path": "/app/models/qwen3-reranker",
            "model_type": "reranker",
        },
        "qwen3-reranker",
    )

    assert captured["torch_dtype"] == "bfloat16"
    assert captured["caller_option"] == "kept"


def test_xinference_rest_launch_maps_decoder_reranker_to_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def launch_model(self, **kwargs) -> None:
            captured.update(kwargs)

    service = deployment_service_module.DeploymentService()
    monkeypatch.setattr(service, "_get_xinference_client", lambda *_args, **_kwargs: _Client())
    monkeypatch.setattr(
        deployment_service_module,
        "_trusted_model_launch_kwargs",
        lambda _config: {},
    )
    deployment = SimpleNamespace(
        xinference_endpoint="http://xinference:9997",
        user_id="user-1",
        config={},
        gpu_id=None,
        replica=1,
        gpu_memory_utilization=0.9,
    )

    service._start_external_deployment(
        deployment,
        {
            "model_name": "Qwen/Qwen3-Reranker-4B",
            "model_path": "/app/models/qwen3-reranker",
            "model_type": "decoder_reranker",
        },
        "qwen3-reranker",
    )

    assert captured["model_type"] == "rerank"
    assert captured["torch_dtype"] == "bfloat16"


def test_xinference_reload_maps_decoder_reranker_to_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def terminate_model(self, _model_uid: str) -> None:
            return None

        def launch_model(self, **kwargs) -> None:
            captured.update(kwargs)

    service = deployment_service_module.DeploymentService()
    monkeypatch.setattr(service, "_get_xinference_client", lambda *_args, **_kwargs: _Client())
    monkeypatch.setattr(
        deployment_service_module.model_registry_service,
        "get_model",
        lambda _model_id: {
            "model_name": "Qwen/Qwen3-Reranker-4B",
            "model_path": "/app/models/qwen3-reranker",
            "model_type": "decoder_reranker",
        },
    )
    monkeypatch.setattr(deployment_service_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        deployment_service_module.docker_deployer,
        "wait_for_model",
        lambda *_args, **_kwargs: True,
    )
    deployment = SimpleNamespace(
        model_uid="qwen3-reranker",
        model_id="model-1",
        xinference_endpoint="http://xinference:9997",
        user_id="user-1",
        config={},
        replica=1,
        gpu_memory_utilization=0.9,
        update_status=lambda _status: None,
    )

    session = SimpleNamespace(add=lambda _value: None, commit=lambda: None)

    assert service._reload_xinference_model(deployment, session) is True
    assert captured["model_type"] == "rerank"
    assert captured["torch_dtype"] == "bfloat16"


def test_xinference_qwen3_embedding_uses_upstream_default_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def launch_model(self, **kwargs) -> None:
            captured.update(kwargs)

    service = deployment_service_module.DeploymentService()
    monkeypatch.setattr(service, "_get_xinference_client", lambda *_args, **_kwargs: _Client())
    monkeypatch.setattr(
        deployment_service_module,
        "_trusted_model_launch_kwargs",
        lambda _config: {"caller_option": "kept"},
    )
    deployment = SimpleNamespace(
        xinference_endpoint="http://xinference:9997",
        user_id="user-1",
        config={},
        gpu_id=None,
        replica=1,
        gpu_memory_utilization=0.9,
    )

    service._start_external_deployment(
        deployment,
        {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "model_path": "/app/models/qwen3-embedding",
            "model_type": "embedding",
        },
        "qwen3-embedding",
    )

    assert "model_engine" not in captured
    assert "torch_dtype" not in captured
    assert captured["caller_option"] == "kept"


@pytest.mark.parametrize(
    ("trusted_framework", "expected_instruction_key"),
    [("vllm", "instruction"), ("sglang", "instruct")],
)
def test_model_config_rerank_proxy_uses_trusted_framework_instruction_field(
    monkeypatch: pytest.MonkeyPatch,
    trusted_framework: str,
    expected_instruction_key: str,
) -> None:
    captured: dict[str, object] = {}

    class _ProxyResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"results": []}

    async def request(method, url, user_id, **kwargs):
        captured.update(
            method=method,
            url=url,
            user_id=user_id,
            **kwargs,
        )
        return _ProxyResponse()

    config = {
        "config_id": "config-1",
        "user_id": "user-1",
        "deployment_id": None,
        "api_endpoint": "https://rerank.example.test/v1",
        "api_key": None,
        "model_type": "reranker",
        "model_name": "Qwen3-Reranker-4B",
        "provider": "local",
        "inference_framework": trusted_framework,
    }
    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "get_config",
        lambda _config_id: config,
    )
    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "update_check_status",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        model_config_routes_module.deployment_service,
        "get_deployment",
        lambda _deployment_id: (_ for _ in ()).throw(
            AssertionError("unbound config must use its trusted framework")
        ),
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "async_request_user_outbound",
        request,
    )
    model_config_routes_module._deployment_framework_cache.clear()

    result = asyncio.run(
        model_config_routes_module.test_api_proxy(
            "config-1",
            model_config_routes_module.TestProxyRequest(
                path="/v1/rerank",
                body={
                    "query": "raw query",
                    "documents": ["raw document"],
                    "instruction": "framework-owned instruction",
                    "instruct": "request-forged-instruct",
                    "inference_framework": (
                        "sglang" if trusted_framework == "vllm" else "vllm"
                    ),
                },
            ),
            current_user={"user_id": "user-1"},
        )
    )

    assert result.success is True
    expected_body = {
        "query": "raw query",
        "documents": ["raw document"],
        expected_instruction_key: "framework-owned instruction",
    }
    if trusted_framework == "vllm":
        expected_body["model"] = "Qwen3-Reranker-4B"
    assert captured["json"] == expected_body


def test_model_config_rerank_proxy_persisted_framework_wins_over_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _ProxyResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"results": []}

    async def request(_method, _url, _user_id, **kwargs):
        captured.update(kwargs)
        return _ProxyResponse()

    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "config-1",
            "user_id": "user-1",
            "deployment_id": "deployment-1",
            "api_endpoint": "https://rerank.example.test/v1",
            "api_key": None,
            "model_type": "reranker",
            "model_name": "Qwen3-Reranker-4B",
            "provider": "local",
            "inference_framework": "vllm",
        },
    )
    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "update_check_status",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "async_request_user_outbound",
        request,
    )
    model_config_routes_module._deployment_framework_cache.clear()
    model_config_routes_module._deployment_framework_cache["deployment-1"] = "sglang"

    result = asyncio.run(
        model_config_routes_module.test_api_proxy(
            "config-1",
            model_config_routes_module.TestProxyRequest(
                path="/v1/rerank",
                body={
                    "query": "raw query",
                    "documents": ["raw document"],
                    "instruction": "trusted instruction",
                },
            ),
            current_user={"user_id": "user-1"},
        )
    )

    assert result.success is True
    assert captured["json"] == {
        "model": "Qwen3-Reranker-4B",
        "query": "raw query",
        "documents": ["raw document"],
        "instruction": "trusted instruction",
    }


def test_model_config_rerank_proxy_falls_back_to_deployment_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _ProxyResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"results": []}

    async def request(_method, _url, _user_id, **kwargs):
        captured.update(kwargs)
        return _ProxyResponse()

    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "config-1",
            "user_id": "user-1",
            "deployment_id": "deployment-1",
            "api_endpoint": "https://rerank.example.test/v1",
            "api_key": None,
            "model_type": "reranker",
            "model_name": "Qwen3-Reranker-4B",
            "provider": "local",
            "inference_framework": None,
        },
    )
    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "update_check_status",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_config_routes_module.deployment_service,
        "get_deployment",
        lambda _deployment_id: {"inference_framework": "sglang"},
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "async_request_user_outbound",
        request,
    )
    model_config_routes_module._deployment_framework_cache.clear()

    result = asyncio.run(
        model_config_routes_module.test_api_proxy(
            "config-1",
            model_config_routes_module.TestProxyRequest(
                path="/v1/rerank",
                body={
                    "query": "raw query",
                    "documents": ["raw document"],
                    "instruction": "trusted instruction",
                },
            ),
            current_user={"user_id": "user-1"},
        )
    )

    assert result.success is True
    assert captured["json"] == {
        "query": "raw query",
        "documents": ["raw document"],
        "instruct": "trusted instruction",
    }
    assert model_config_routes_module._deployment_framework_cache == {
        "deployment-1": "sglang"
    }


@pytest.mark.parametrize("trusted_framework", ["vllm", "sglang"])
@pytest.mark.parametrize("instruction", [None, "", "   "])
def test_model_config_rerank_proxy_omits_blank_instruction(
    monkeypatch: pytest.MonkeyPatch,
    trusted_framework: str,
    instruction: str | None,
) -> None:
    captured: dict[str, object] = {}

    class _ProxyResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"results": []}

    async def request(_method, _url, _user_id, **kwargs):
        captured.update(kwargs)
        return _ProxyResponse()

    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "get_config",
        lambda _config_id: {
            "config_id": "config-1",
            "user_id": "user-1",
            "deployment_id": None,
            "api_endpoint": "https://rerank.example.test/v1",
            "api_key": None,
            "model_type": "reranker",
            "model_name": "reranker",
            "provider": "local",
            "inference_framework": trusted_framework,
        },
    )
    monkeypatch.setattr(
        model_config_routes_module.model_config_service,
        "update_check_status",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "validate_user_outbound_url",
        lambda endpoint, _user_id: endpoint,
    )
    monkeypatch.setattr(
        model_config_routes_module,
        "async_request_user_outbound",
        request,
    )
    model_config_routes_module._deployment_framework_cache.clear()

    result = asyncio.run(
        model_config_routes_module.test_api_proxy(
            "config-1",
            model_config_routes_module.TestProxyRequest(
                path="/v1/rerank",
                body={
                    "query": "raw query",
                    "documents": ["raw document"],
                    "instruction": instruction,
                },
            ),
            current_user={"user_id": "user-1"},
        )
    )

    assert result.success is True
    assert "instruction" not in captured["json"]
    assert "instruct" not in captured["json"]
