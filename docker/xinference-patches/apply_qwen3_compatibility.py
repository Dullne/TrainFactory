from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import Any


TARGET_PAYLOADS = {
    Path("model/embedding/sentence_transformers/core.py"): "embedding",
    Path("model/rerank/sentence_transformers/core.py"): "rerank",
}

TRANSFORMATIONS = {
    Path("model/embedding/sentence_transformers/core.py"): (
        (
            "    allow_trust_remote_code,\n",
            "",
        ),
        (
            "logger = logging.getLogger(__name__)\n"
            "SENTENCE_TRANSFORMER_MODEL_LIST: List[str] = []\n",
            "logger = logging.getLogger(__name__)\n"
            "SENTENCE_TRANSFORMER_MODEL_LIST: List[str] = []\n"
            "ALLOW_MODEL_REMOTE_CODE = os.environ.get(\n"
            '    "ALLOW_MODEL_REMOTE_CODE", "false"\n'
            ').strip().lower() in {"1", "true", "yes", "on"}\n',
        ),
        (
            "            if not allow_trust_remote_code(self.model_family):\n",
            "            if not ALLOW_MODEL_REMOTE_CODE:\n",
        ),
        (
            '                    "repository; set XINFERENCE_TRUST_REMOTE_CODE=1 to allow it."\n',
            '                    "repository; set ALLOW_MODEL_REMOTE_CODE=true to allow it."\n',
        ),
        (
            "                tokenizer_kwargs=tokenizer_kwargs,\n"
            "                truncate_dim=dimensions,\n",
            "                tokenizer_kwargs=tokenizer_kwargs,\n"
            "                trust_remote_code=ALLOW_MODEL_REMOTE_CODE,\n"
            "                truncate_dim=dimensions,\n",
        ),
        (
            "                truncate_dim=dimensions,\n"
            "            )\n"
            "        else:\n",
            "                truncate_dim=dimensions,\n"
            "            )\n"
            "            # Preserve the Qwen3 last-token pooling contract.\n"
            "            for module in self._model:\n"
            '                if hasattr(module, "pooling_mode_mean_tokens"):\n'
            "                    module.pooling_mode_mean_tokens = False\n"
            "                    module.pooling_mode_lasttoken = True\n"
            "        else:\n",
        ),
        (
            "                trust_remote_code=allow_trust_remote_code(self.model_family),\n",
            "                trust_remote_code=ALLOW_MODEL_REMOTE_CODE,\n",
        ),
    ),
    Path("model/rerank/sentence_transformers/core.py"): (
        (
            "    allow_trust_remote_code,\n",
            "",
        ),
        (
            "logger = logging.getLogger(__name__)\n",
            "logger = logging.getLogger(__name__)\n"
            "ALLOW_MODEL_REMOTE_CODE = os.environ.get(\n"
            '    "ALLOW_MODEL_REMOTE_CODE", "false"\n'
            ').strip().lower() in {"1", "true", "yes", "on"}\n'
            "\n"
            "\n"
            "def _sanitize_remote_code_kwargs(value):\n"
            "    if isinstance(value, dict):\n"
            "        return {\n"
            "            key: _sanitize_remote_code_kwargs(item)\n"
            "            for key, item in value.items()\n"
            "            if not (\n"
            "                isinstance(key, str)\n"
            '                and key.strip().lower().replace("-", "_") == "trust_remote_code"\n'
            "            )\n"
            "        }\n"
            "    if isinstance(value, list):\n"
            "        return [_sanitize_remote_code_kwargs(item) for item in value]\n"
            "    return value\n",
        ),
        (
            "            if not allow_trust_remote_code(self.model_family):\n",
            "            if not ALLOW_MODEL_REMOTE_CODE:\n",
        ),
        (
            '                    "repository; set XINFERENCE_TRUST_REMOTE_CODE=1 to allow it."\n',
            '                    "repository; set ALLOW_MODEL_REMOTE_CODE=true to allow it."\n',
        ),
        (
            "            self._model = CrossEncoder(\n"
            "                self._model_path,\n"
            "                device=self._device,\n"
            "                trust_remote_code=allow_trust_remote_code(self.model_family),\n"
            '                max_length=getattr(self.model_family, "max_tokens"),\n'
            "                **self._kwargs,\n"
            "            )\n",
            "            cross_encoder_kwargs = _sanitize_remote_code_kwargs(self._kwargs)\n"
            "            self._model = CrossEncoder(\n"
            "                self._model_path,\n"
            "                device=self._device,\n"
            "                trust_remote_code=ALLOW_MODEL_REMOTE_CODE,\n"
            '                max_length=getattr(self.model_family, "max_tokens"),\n'
            "                **cross_encoder_kwargs,\n"
            "            )\n",
        ),
        (
            "            tokenizer = AutoTokenizer.from_pretrained(\n"
            '                self._model_path, padding_side="left"\n'
            "            )\n",
            "            tokenizer = AutoTokenizer.from_pretrained(\n"
            "                self._model_path,\n"
            '                padding_side="left",\n'
            "                trust_remote_code=ALLOW_MODEL_REMOTE_CODE,\n"
            "            )\n",
        ),
        (
            '                model_kwargs["torch_dtype"] = torch.float16\n'
            "            model_kwargs.update(self._kwargs)\n",
            '                model_kwargs["torch_dtype"] = (\n'
            '                    torch.bfloat16\n'
            '                    if "qwen3" in self.model_family.model_name.lower()\n'
            '                    else torch.float16\n'
            '                )\n'
            "            model_kwargs.update(_sanitize_remote_code_kwargs(self._kwargs))\n"
            '            model_kwargs["trust_remote_code"] = ALLOW_MODEL_REMOTE_CODE\n',
        ),
        (
            "                scores = batch_scores[:, 1].exp().tolist()\n",
            "                true_probabilities = batch_scores[:, 1]\n"
            '                if "qwen3" in self.model_family.model_name.lower():\n'
            "                    true_probabilities = true_probabilities.float()\n"
            "                scores = true_probabilities.exp().tolist()\n",
        ),
    ),
}


def _canonical_source(source: bytes) -> bytes:
    return source.replace(b"\r\n", b"\n")


def _sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _apply_transformations(relative_path: Path, source: bytes) -> bytes:
    patched = source
    for old_text, new_text in TRANSFORMATIONS[relative_path]:
        old = old_text.encode("utf-8")
        if patched.count(old) != 1:
            raise RuntimeError(
                f"compatibility patch anchor mismatch: {relative_path.as_posix()}"
            )
        patched = patched.replace(old, new_text.encode("utf-8"), 1)
    return patched


def apply_compatibility_payloads(
    *,
    package_root: Path,
    payload_root: Path,
    contract: dict[str, Any],
) -> None:
    xinference_contract = contract["xinference"]
    for relative_path, source_key in TARGET_PAYLOADS.items():
        source_contract = xinference_contract["sources"][source_key]
        contract_runtime_path = Path(source_contract["runtime_path"])
        if contract_runtime_path != relative_path:
            raise RuntimeError(f"runtime path mismatch: {source_key}")
        target_path = package_root / contract_runtime_path
        payload_name = Path(source_contract["vendored_path"]).name
        payload_path = payload_root / payload_name
        target_source = _canonical_source(target_path.read_bytes())
        snapshot_source = _canonical_source(payload_path.read_bytes())
        target_hash = _sha256(target_source)
        upstream_hash = source_contract["sha256"]
        patched_hash = source_contract["patched_sha256"]

        if _sha256(snapshot_source) != upstream_hash:
            raise RuntimeError(f"upstream source snapshot hash mismatch: {payload_name}")

        if target_hash == patched_hash:
            continue
        if target_hash != upstream_hash:
            raise RuntimeError(f"preimage hash mismatch: {relative_path.as_posix()}")

        patched_source = _apply_transformations(relative_path, target_source)
        if _sha256(patched_source) != patched_hash:
            raise RuntimeError(f"patched source hash mismatch: {relative_path.as_posix()}")
        compile(patched_source, os.fspath(target_path), "exec")

        temporary_path = target_path.with_name(f".{target_path.name}.trainfactory.tmp")
        temporary_path.write_bytes(patched_source)
        os.chmod(temporary_path, stat.S_IMODE(target_path.stat().st_mode))
        os.replace(temporary_path, target_path)


def _find_xinference_package_root() -> Path:
    spec = importlib.util.find_spec("xinference")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("xinference package is not installed")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _verify_installed_distribution(contract: dict[str, Any]) -> None:
    distribution = importlib.metadata.distribution("xinference")
    expected = contract["xinference"]
    if distribution.version != expected["installed_version"]:
        raise RuntimeError("xinference installed version mismatch")

    metadata_text = distribution.read_text("METADATA")
    if metadata_text is None:
        raise RuntimeError("xinference METADATA is missing")
    if _sha256(metadata_text.encode("utf-8")) != expected[
        "installed_metadata_sha256"
    ]:
        raise RuntimeError("xinference installed metadata hash mismatch")


def _verify_configured_image(contract: dict[str, Any], image_ref: str) -> None:
    if image_ref != contract["xinference"]["image"]:
        raise RuntimeError("xinference image reference mismatch")


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 2:
        raise RuntimeError("unsupported inference compatibility contract")
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--payload-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    contract = _load_contract(args.contract)
    _verify_configured_image(contract, args.image_ref)
    if args.package_root is None:
        _verify_installed_distribution(contract)
        package_root = _find_xinference_package_root()
    else:
        package_root = args.package_root
    apply_compatibility_payloads(
        package_root=package_root,
        payload_root=args.payload_root,
        contract=contract,
    )

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if command:
        os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
