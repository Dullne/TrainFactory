"""Local filesystem storage backend."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import StorageBackend

logger = logging.getLogger(__name__)


def _dir_size(dir_path: Path) -> int:
    """Calculate total size of all files in a directory."""
    total = 0
    for root, _, files in os.walk(dir_path):
        for file in files:
            try:
                total += (Path(root) / file).stat().st_size
            except OSError:
                continue
    return total


def _is_arrow_dataset_dir(dir_path: Path) -> bool:
    """Check if a directory is an Arrow dataset (HuggingFace datasets format)."""
    return dir_path.is_dir() and (
        (dir_path / "dataset_info.json").exists() or any(dir_path.glob("*.arrow"))
    )


def _count_file_rows(file_path: Path, file_format: str) -> int:
    """Count rows in a single file."""
    try:
        if file_format == "jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        elif file_format == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 0
        elif file_format == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(file_path).num_rows
        elif file_format == "arrow":
            from datasets import Dataset

            ds = Dataset.load_from_disk(
                str(file_path.parent if file_path.suffix == ".arrow" else file_path)
            )
            return len(ds)
        elif file_format == "csv":
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1  # Minus header
    except Exception:
        return 0
    return 0


def _load_single_file_preview(
    file_path: Path, file_format: str, limit: int
) -> Dict[str, Any]:
    """Load preview from a single file."""
    rows: List[Dict[str, Any]] = []
    columns: List[Dict[str, Any]] = []
    num_rows = 0

    try:
        if file_format == "jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        # Gate on collected-row count, not physical line index, so
                        # blank lines within the first `limit` lines don't drop rows.
                        if len(rows) < limit:
                            row = json.loads(line)
                            rows.append(row)
                            if not columns:
                                columns = [
                                    {"name": k, "type": type(v).__name__}
                                    for k, v in row.items()
                                ]
                        num_rows += 1

        elif file_format == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows = data[:limit]
                    num_rows = len(data)
                    if rows:
                        columns = [
                            {"name": k, "type": type(v).__name__}
                            for k, v in rows[0].items()
                        ]

        elif file_format == "parquet":
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(file_path)
                num_rows = table.num_rows
                columns = [
                    {"name": col, "type": str(table.schema.field(col).type)}
                    for col in table.column_names
                ]
                rows = table.slice(0, limit).to_pylist()
            except ImportError:
                return {"error": "pyarrow not installed for parquet support"}

        elif file_format == "csv":
            import csv

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < limit:
                        rows.append(dict(row))
                    num_rows += 1
                if rows:
                    columns = [{"name": k, "type": "str"} for k in rows[0].keys()]

        elif file_format == "arrow":
            try:
                from datasets import Dataset

                ds = Dataset.load_from_disk(
                    str(file_path.parent if file_path.suffix == ".arrow" else file_path)
                )
                num_rows = len(ds)
                columns = (
                    [
                        {"name": name, "type": str(feature)}
                        for name, feature in ds.features.items()
                    ]
                    if ds.features
                    else []
                )
                if num_rows:
                    sample = ds[: min(limit, num_rows)]
                    if sample:
                        keys = list(sample.keys())
                        rows = [
                            dict(zip(keys, values))
                            for values in zip(*sample.values())
                        ]
            except ImportError:
                return {"error": "datasets not installed for arrow support"}

        return {"columns": columns, "rows": rows, "num_rows": num_rows}

    except Exception as e:
        logger.error(f"Error loading file preview: {e}")
        return {"error": str(e)}


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend for dataset operations."""

    def exists(self, reference: str) -> bool:
        return Path(reference).exists()

    def delete(self, reference: str, dataset_id: Optional[str] = None) -> bool:
        """Delete local file/dir only under trusted managed roots.

        First-level directories under a trusted root are blocked to avoid broad
        deletes. ``dataset_id`` can whitelist one exact directory name (e.g.
        downloaded dataset_id folders).
        """
        if not reference:
            return True

        from ...config.settings import get_settings

        settings = get_settings()
        trusted_roots = {
            Path(settings.datasets_dir).resolve(),
            Path(settings.local_cache_dir).resolve(),
            Path(
                os.environ.get(
                    "GENERATION_OUTPUT_DIR", str(settings.datasets_dir)
                )
            ).resolve(),
            Path(os.environ.get("SYNC_DATA_DIR", "/app/data/sync")).resolve(),
        }

        try:
            target = Path(reference).resolve()
        except Exception:
            logger.warning(f"Refused to delete invalid path: {reference!r}")
            return False

        matched_root = None
        for root in trusted_roots:
            try:
                rel = target.relative_to(root)
                matched_root = (root, rel)
                break
            except ValueError:
                continue

        if matched_root is None:
            logger.warning(
                "Refused to delete path outside trusted roots: %s", target
            )
            return False

        _, rel = matched_root
        if not rel.parts:
            logger.warning("Refused to delete trusted root: %s", target)
            return False

        if target.is_dir() and len(rel.parts) < 2:
            if not (
                len(rel.parts) == 1
                and dataset_id
                and rel.parts[0] == dataset_id
            ):
                logger.warning(
                    "Refused to recursively delete broad directory: %s", target
                )
                return False

        try:
            if target.is_file():
                os.remove(target)
                logger.info(f"Deleted file: {target}")
            elif target.is_dir():
                shutil.rmtree(target)
                logger.info(f"Deleted directory: {target}")
            return True
        except OSError as e:
            logger.warning(f"Failed to delete {target}: {e}")
            return False

    def load_preview(
        self, reference: str, file_format: str, limit: int = 10
    ) -> Dict[str, Any]:
        """Load preview data from a local dataset file or directory."""
        path = Path(reference)
        result: Dict[str, Any] = {
            "columns": [],
            "rows": [],
            "num_rows": 0,
            "num_train": None,
            "num_eval": None,
            "num_test": None,
            "file_size": 0,
        }

        # Single file
        if path.is_file():
            file_format = self.detect_file_format(path)
            result["file_size"] = path.stat().st_size
            preview = _load_single_file_preview(path, file_format, limit)
            result.update(preview)
            return result

        if not path.is_dir():
            return {"error": f"Path not found: {reference}"}

        # --- Directory: look for Arrow split dirs first ---
        train_file: Optional[Path] = None
        val_file: Optional[Path] = None
        test_file: Optional[Path] = None
        total_size = 0

        arrow_splits: Dict[str, Path] = {}
        for split_name in ["train", "val", "validation", "dev", "test"]:
            split_dir = path / split_name
            if _is_arrow_dataset_dir(split_dir):
                arrow_splits[split_name] = split_dir

        if arrow_splits:
            train_file = arrow_splits.get("train")
            val_file = (
                arrow_splits.get("val")
                or arrow_splits.get("validation")
                or arrow_splits.get("dev")
            )
            test_file = arrow_splits.get("test")

            if train_file:
                result["num_train"] = _count_file_rows(train_file, "arrow")
                total_size += _dir_size(train_file)
            if val_file:
                result["num_eval"] = _count_file_rows(val_file, "arrow")
                total_size += _dir_size(val_file)
            if test_file:
                result["num_test"] = _count_file_rows(test_file, "arrow")
                total_size += _dir_size(test_file)

            result["file_size"] = total_size
            result["num_rows"] = (
                (result["num_train"] or 0)
                + (result["num_eval"] or 0)
                + (result["num_test"] or 0)
            )

            preview_file = train_file or val_file or test_file
            preview = _load_single_file_preview(preview_file, "arrow", limit)
            result["columns"] = preview.get("columns", [])
            result["rows"] = preview.get("rows", [])
            return result

        # --- Directory: look for split files ---
        train_patterns = [
            "train.jsonl", "train.json", "train.parquet", "train.csv", "train.arrow",
        ]
        val_patterns = [
            "eval.jsonl", "eval.json", "eval.parquet", "eval.csv", "eval.arrow",
            "val.jsonl", "val.json", "validation.jsonl", "validation.json",
            "dev.jsonl", "dev.json",
            "val.parquet", "validation.parquet", "val.csv", "validation.csv",
            "val.arrow", "validation.arrow", "dev.arrow",
        ]
        test_patterns = [
            "test.jsonl", "test.json", "test.parquet", "test.csv", "test.arrow",
        ]
        data_patterns = [
            "data.jsonl", "data.json", "data.parquet", "data.csv", "data.arrow",
        ]

        for pattern in train_patterns:
            files = list(path.glob(pattern))
            if files:
                train_file = files[0]
                break

        for pattern in val_patterns:
            files = list(path.glob(pattern))
            if files:
                val_file = files[0]
                break

        for pattern in test_patterns:
            files = list(path.glob(pattern))
            if files:
                test_file = files[0]
                break

        if not train_file:
            # Exclude files already claimed as val/test so the generic glob does
            # not re-select the same eval.jsonl as the train file and double-count
            # num_rows / file_size for single-file eval/test datasets.
            already = {f for f in (val_file, test_file) if f}
            for pattern in data_patterns + [
                "*.jsonl", "*.json", "*.parquet", "*.csv", "*.arrow",
            ]:
                files = [f for f in path.glob(pattern) if f not in already]
                if files:
                    train_file = files[0]
                    break

        if not (train_file or val_file or test_file):
            return {"error": f"No data file found in directory: {reference}"}

        if train_file:
            fmt = self.detect_file_format(train_file)
            result["num_train"] = _count_file_rows(train_file, fmt)
            total_size += train_file.stat().st_size

        if val_file:
            fmt = self.detect_file_format(val_file)
            result["num_eval"] = _count_file_rows(val_file, fmt)
            total_size += val_file.stat().st_size

        if test_file:
            fmt = self.detect_file_format(test_file)
            result["num_test"] = _count_file_rows(test_file, fmt)
            total_size += test_file.stat().st_size

        result["file_size"] = total_size
        result["num_rows"] = (
            (result["num_train"] or 0)
            + (result["num_eval"] or 0)
            + (result["num_test"] or 0)
        )

        # Prefer train for preview, but fall back to val/test so an eval/test-only
        # directory (train_file is None) does not crash detect_file_format(None).
        preview_file = train_file or val_file or test_file
        preview = _load_single_file_preview(
            preview_file, self.detect_file_format(preview_file), limit
        )
        result["columns"] = preview.get("columns", [])
        result["rows"] = preview.get("rows", [])

        return result

    def resolve_export_source(
        self, reference: str, declared_format: str
    ) -> Tuple[Path, str]:
        """Resolve the concrete local source file/dir for export."""
        path = Path(reference)
        if path.is_file():
            return path, self.detect_file_format(path)

        if not path.is_dir():
            raise ValueError(f"Path not found: {reference}")

        if declared_format == "arrow":
            split_train = path / "train"
            if split_train.exists() and split_train.is_dir():
                return split_train, "arrow"
            return path, "arrow"

        patterns = [
            "train.jsonl", "train.json", "train.csv", "train.parquet",
            "data.jsonl", "data.json", "data.csv", "data.parquet",
            "*.jsonl", "*.json", "*.csv", "*.parquet",
        ]
        for pattern in patterns:
            files = sorted(path.glob(pattern))
            if files:
                source = files[0]
                return source, self.detect_file_format(source)

        raise ValueError(f"No exportable file found under: {reference}")
