"""
Unified data loader for TrainFactory.

Supports loading datasets from:
- HuggingFace Hub
- ModelScope
- Local JSON/JSONL/Parquet/CSV files
- Local directories with standard naming (including Arrow split dirs)
- S3/MinIO object storage URIs (auto-downloaded to local cache)
"""

import hashlib
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from uuid import uuid4

from datasets import load_dataset, Dataset

from ..config import settings

logger = logging.getLogger(__name__)

_S3_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_S3_CACHE_LOCKS_GUARD = threading.Lock()

# Format to HF loader type mapping
_FORMAT_LOADER = {
    ".jsonl": "json",
    ".json": "json",
    ".parquet": "parquet",
    ".csv": "csv",
}


class DataLoader:
    """Unified data loader, training-type agnostic."""

    DEFAULT_HF_SUBSET = "pair-class"

    def __init__(
        self,
        hf_subset: Optional[str] = None,
        train_sample_size: int = -1,
        eval_sample_size: int = -1,
        test_sample_size: int = -1
    ):
        """
        Initialize data loader.

        Args:
            hf_subset: HuggingFace dataset subset name, supports comma-separated list
            train_sample_size: Train dataset sample limit, -1 for no limit, 0 to skip
            eval_sample_size: Eval dataset sample limit, -1 for no limit, 0 to skip
            test_sample_size: Test dataset sample limit, -1 for no limit, 0 to skip
        """
        # Parse HF_subset to list
        if hf_subset and isinstance(hf_subset, str):
            if "," in hf_subset:
                parts = [s.strip() for s in hf_subset.split(",")]
                self.hf_subsets = []
                for part in parts:
                    if part == "" or part.lower() in ["none", "null"]:
                        self.hf_subsets.append(None)
                    else:
                        self.hf_subsets.append(part)
            else:
                if hf_subset.lower() in ["none", "null"]:
                    self.hf_subsets = [None]
                else:
                    self.hf_subsets = [hf_subset]
        elif isinstance(hf_subset, list):
            self.hf_subsets = hf_subset
        else:
            self.hf_subsets = []

        self.hf_subset = hf_subset
        self.train_sample_size = max(train_sample_size, -1)
        self.eval_sample_size = max(eval_sample_size, -1)
        self.test_sample_size = max(test_sample_size, -1)

        # Configure HuggingFace cache directory
        self.hf_cache_dir = settings.hf_cache_dir

        logger.info(f"DataLoader initialized: hf_subset={hf_subset}, "
                   f"train_sample={self.train_sample_size}, "
                   f"eval_sample={self.eval_sample_size}, "
                   f"test_sample={self.test_sample_size}")

    def load_all_splits(
        self,
        dataset_path: str
    ) -> Tuple[Union[Dataset, Dict[str, Dataset], None], ...]:
        """
        Load all data splits in parallel.

        Args:
            dataset_path: Dataset path (HuggingFace name, local file, or directory)

        Returns:
            Tuple of (train_dataset, eval_dataset, test_dataset)
        """
        splits = ["train", "eval", "test"]
        results: Dict[str, Union[Dataset, Dict[str, Dataset], None]] = {s: None for s in splits}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.load_data, split, dataset_path): split
                for split in splits
            }
            for future in as_completed(futures):
                split = futures[future]
                try:
                    results[split] = future.result()
                except Exception as e:
                    if split == "train":
                        raise
                    logger.warning(f"Failed to load {split} split: {e}")
                    results[split] = None

        return results["train"], results["eval"], results["test"]

    def load_data(
        self,
        split_type: str,
        dataset_path: str
    ) -> Union[Optional[Dataset], Dict[str, Dataset]]:
        """
        Load dataset for a specific split.

        Args:
            split_type: Dataset split type ('train', 'eval', 'test')
            dataset_path: Dataset path

        Returns:
            Dataset, Dict[str, Dataset], or None
        """
        if not dataset_path:
            logger.error("Dataset path not provided")
            return None

        return self._load_from_unified_path(dataset_path, split_type)

    def _is_huggingface_dataset(self, dataset_path: str) -> bool:
        """Check if path is a HuggingFace dataset name."""
        path = dataset_path.strip()

        # S3 URI is not a HuggingFace dataset
        if path.startswith("s3://"):
            return False

        # Check for file extensions (local file indicator)
        if path.endswith(('.json', '.jsonl', '.txt', '.csv', '.parquet')):
            return False

        # Check for relative paths
        if path.startswith('./') or path.startswith('../'):
            return False

        # Check for HuggingFace format: org/repo
        if '/' in path:
            parts = path.split('/')
            if len(parts) == 2 and parts[0] and parts[1]:
                return True
            elif len(parts) > 2:
                return False

        # Check for absolute path
        if os.path.isabs(path):
            return False

        # Check if exists locally
        if os.path.exists(path):
            return False

        # Assume HuggingFace dataset if no local file indicators
        return True

    def _load_from_unified_path(
        self,
        dataset_path: str,
        split_type: str
    ) -> Union[Optional[Dataset], Dict[str, Dataset]]:
        """Load from unified path."""
        # Handle comma-separated multiple paths
        if "," in dataset_path:
            return self._load_from_mixed_sources(dataset_path, split_type)

        # Resolve S3 URIs to local cache
        if dataset_path.startswith("s3://"):
            dataset_path = self._resolve_s3_to_local(dataset_path)

        # Single path handling
        if self._is_huggingface_dataset(dataset_path):
            logger.info(f"Identified as HuggingFace dataset: {dataset_path}")
            return self._load_from_hub(dataset_path, split_type)
        else:
            logger.info(f"Identified as local path: {dataset_path}")
            return self._load_from_local(dataset_path, split_type)

    def _resolve_s3_to_local(self, s3_uri: str) -> str:
        """Download S3 dataset to local cache, return cached local path.

        Uses content-addressable caching: hash(s3_uri) -> local_cache_dir/{hash}/
        Skips download if already cached.

        Downloads to a temporary directory first, then atomically renames to
        the target cache directory on success. This prevents incomplete cache
        from being used and handles concurrent downloads safely.
        """
        import shutil

        def _resolve_cached_target(cache_root: Path) -> str:
            marker = cache_root / ".single_object"
            if marker.exists():
                try:
                    rel_path = marker.read_text(encoding="utf-8").strip()
                except OSError:
                    rel_path = ""
                if rel_path:
                    candidate = cache_root / rel_path
                    if candidate.exists():
                        return str(candidate)

            return str(cache_root)

        cache_key = hashlib.sha256(s3_uri.encode()).hexdigest()[:16]
        cache_dir = Path(settings.local_cache_dir) / cache_key

        if cache_dir.exists() and any(cache_dir.iterdir()):
            resolved_path = _resolve_cached_target(cache_dir)
            logger.info(f"Using cached S3 dataset: {s3_uri} -> {resolved_path}")
            return resolved_path

        # Per-key in-process lock prevents parallel split loads from racing
        # on cache promotion/removal.
        with _S3_CACHE_LOCKS_GUARD:
            lock = _S3_CACHE_LOCKS.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                _S3_CACHE_LOCKS[cache_key] = lock

        with lock:
            if cache_dir.exists() and any(cache_dir.iterdir()):
                resolved_path = _resolve_cached_target(cache_dir)
                logger.info(f"Using cached S3 dataset: {s3_uri} -> {resolved_path}")
                return resolved_path

            # Use per-call temp directory to avoid cross-thread deletion races.
            tmp_dir = Path(settings.local_cache_dir) / (
                f"{cache_key}.downloading.{os.getpid()}.{uuid4().hex}"
            )
            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)

                from ..storage.object_store import get_object_store, uri_to_key

                store = get_object_store()
                key = uri_to_key(s3_uri, expected_bucket=store.bucket)

                objects = store.list_objects(key)
                single_rel_path: Optional[str] = None
                if not objects:
                    local_file = tmp_dir / Path(key).name
                    store.download_file(key, str(local_file))
                    single_rel_path = local_file.relative_to(tmp_dir).as_posix()
                elif len(objects) == 1 and objects[0].key == key:
                    local_file = tmp_dir / Path(key).name
                    store.download_file(key, str(local_file))
                    single_rel_path = local_file.relative_to(tmp_dir).as_posix()
                else:
                    for obj in objects:
                        rel = obj.key[len(key):].lstrip("/")
                        if not rel:
                            rel = Path(obj.key).name
                        local_path = tmp_dir / rel
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        store.download_file(obj.key, str(local_path))
                if single_rel_path:
                    (tmp_dir / ".single_object").write_text(single_rel_path, encoding="utf-8")

                # Atomic rename: if another worker already populated cache_dir,
                # discard this temp dir and use the existing cache.
                try:
                    tmp_dir.rename(cache_dir)
                except OSError:
                    if cache_dir.exists() and any(cache_dir.iterdir()):
                        logger.info(f"Cache populated by another process: {cache_dir}")
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    else:
                        raise

                resolved_path = _resolve_cached_target(cache_dir)
                logger.info(f"Downloaded S3 dataset to cache: {s3_uri} -> {resolved_path}")
                return resolved_path

            except Exception:
                # Clean up incomplete temp directory on any failure
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

    def _load_from_hub(
        self,
        dataset_name: str,
        split_type: str
    ) -> Optional[Dataset]:
        """Load from HuggingFace Hub or ModelScope."""
        logger.info(f"Loading from Hub: {dataset_name}, split: {split_type}")

        # Determine subset config
        if self.hf_subsets:
            subset = self.hf_subsets[0]
            config_alternatives = [subset, "pair-class", "pair-score", None]
        else:
            config_alternatives = ["pair-class", "pair-score", None]

        # Split name alternatives
        split_alternatives = [split_type]
        if split_type == "eval":
            split_alternatives = ["eval", "dev", "validation"]
        elif split_type == "dev":
            split_alternatives = ["dev", "eval", "validation"]

        # ModelScope 失败一次后跳过后续组合的重复慢试（否则每个 split/config
        # 组合都先做一次 ModelScope 网络往返，加载慢 2-6 倍）
        modelscope_unavailable = False
        for split_name in split_alternatives:
            for config in config_alternatives:
                try:
                    # Try ModelScope first
                    if not modelscope_unavailable:
                        try:
                            from modelscope.msdatasets import MsDataset
                            logger.info(f"Trying ModelScope: {dataset_name}")

                            if config is None:
                                ms_dataset = MsDataset.load(dataset_name, split=split_name)
                            else:
                                ms_dataset = MsDataset.load(dataset_name, subset_name=config, split=split_name)

                            dataset = Dataset.from_dict(ms_dataset.to_dict())
                            logger.info(f"Successfully loaded from ModelScope: {dataset_name}")
                            return self._apply_sample_size(dataset, split_type)

                        except ImportError:
                            logger.debug("ModelScope not installed, falling back to HuggingFace")
                            modelscope_unavailable = True
                        except Exception as ms_error:
                            logger.debug(f"ModelScope failed: {ms_error}, trying HuggingFace")
                            modelscope_unavailable = True

                    # Fall back to HuggingFace
                    if config is None:
                        dataset = load_dataset(dataset_name, split=split_name, cache_dir=self.hf_cache_dir)
                    else:
                        dataset = load_dataset(dataset_name, config, split=split_name, cache_dir=self.hf_cache_dir)

                    logger.info(f"Successfully loaded from HuggingFace: {dataset_name}")
                    return self._apply_sample_size(dataset, split_type)

                except Exception as e:
                    if "Unknown config" in str(e) or "Config name is missing" in str(e):
                        continue
                    if "Unknown split" in str(e):
                        break
                    logger.debug(f"Failed to load with config={config}, split={split_name}: {e}")

        logger.warning(f"Failed to load dataset: {dataset_name}, split={split_type}")
        return None

    def _load_from_local(
        self,
        path: str,
        split_type: str
    ) -> Optional[Dataset]:
        """Load from local path."""
        if os.path.isdir(path):
            return self._load_from_directory(path, split_type)
        elif os.path.isfile(path):
            filename = Path(path).name.lower()

            if split_type == "train":
                return self._load_file(path, split_type)

            split_prefixes = {
                "eval": ("eval", "val", "validation", "dev"),
                "dev": ("dev", "eval", "val", "validation"),
                "test": ("test",),
            }.get(split_type, (split_type,))

            if any(
                filename == prefix
                or filename.startswith(f"{prefix}.")
                or filename.startswith(f"{prefix}_")
                for prefix in split_prefixes
            ):
                return self._load_file(path, split_type)

            logger.info(
                "Single file path %s does not match split %s, returning None",
                path,
                split_type,
            )
            return None
        else:
            logger.error(f"Path does not exist: {path}")
            if split_type == "train":
                raise ValueError(f"Training data path does not exist: {path}")
            return None

    def _load_from_directory(
        self,
        dir_path: str,
        split_type: str
    ) -> Optional[Dataset]:
        """Load from directory with standard file naming.

        Checks Arrow split directories first (zero-copy mmap), then
        standard file patterns for all supported formats.
        """
        # 1. Check for Arrow split directories (fastest: mmap zero-copy)
        arrow_split_names = {
            'train': ['train'],
            'eval': ['eval', 'val', 'validation', 'dev'],
            'dev': ['dev', 'eval', 'val', 'validation'],
            'test': ['test'],
        }
        for name in arrow_split_names.get(split_type, [split_type]):
            arrow_dir = os.path.join(dir_path, name)
            if self._is_arrow_dataset_dir(arrow_dir):
                logger.info(f"Loading Arrow dataset: {arrow_dir}")
                try:
                    dataset = Dataset.load_from_disk(arrow_dir)
                    return self._apply_sample_size(dataset, split_type)
                except Exception as e:
                    logger.warning(f"Failed to load Arrow dir {arrow_dir}: {e}")

        # 2. Check standard file patterns (all supported formats)
        standard_files = {
            'train': [
                'train_data.jsonl', 'train_data.json', 'train.jsonl', 'train.json',
                'train.parquet', 'train.csv',
            ],
            'eval': [
                'eval_data.jsonl', 'eval_data.json', 'eval.jsonl', 'eval.json',
                'val_data.jsonl', 'val.jsonl', 'dev_data.jsonl', 'dev.jsonl',
                'eval.parquet', 'val.parquet', 'validation.parquet',
                'eval.csv', 'val.csv',
            ],
            'dev': [
                'eval_data.jsonl', 'eval_data.json', 'dev_data.jsonl', 'dev.jsonl',
                'dev.parquet', 'dev.csv',
            ],
            'test': [
                'test_data.jsonl', 'test_data.json', 'test.jsonl', 'test.json',
                'test.parquet', 'test.csv',
            ],
        }

        possible_names = standard_files.get(split_type, [])
        for name in possible_names:
            file_path = os.path.join(dir_path, name)
            if os.path.exists(file_path):
                logger.info(f"Loading standard file: {file_path}")
                try:
                    return self._load_file(file_path, split_type)
                except Exception as e:
                    logger.error(f"Failed to load file {file_path}: {e}")
                    if split_type == "train":
                        raise ValueError(f"Failed to load training data: {e}")
                    return None

        logger.info(f"No standard file found for {split_type} in {dir_path}")
        return None

    def _load_file(self, file_path: str, split_type: str) -> Optional[Dataset]:
        """Load a single data file. Supports JSONL, JSON, Parquet, CSV."""
        ext = os.path.splitext(file_path)[1].lower()
        loader_type = _FORMAT_LOADER.get(ext)

        if loader_type:
            logger.info(f"Loading {ext} file: {file_path}")
            dataset = load_dataset(
                loader_type, data_files=file_path, split="train",
                cache_dir=self.hf_cache_dir,
            )
            return self._apply_sample_size(dataset, split_type)

        # Fallback: try as JSON
        logger.info(f"Loading file as JSON (unknown ext {ext}): {file_path}")
        dataset = load_dataset(
            "json", data_files=file_path, split="train",
            cache_dir=self.hf_cache_dir,
        )
        return self._apply_sample_size(dataset, split_type)

    @staticmethod
    def _is_arrow_dataset_dir(path: str) -> bool:
        """Check if a directory is a HuggingFace Arrow dataset."""
        if not os.path.isdir(path):
            return False
        return (
            os.path.exists(os.path.join(path, "dataset_info.json"))
            or any(Path(path).glob("*.arrow"))
        )

    def _load_from_mixed_sources(
        self,
        mixed_paths: str,
        split_type: str
    ) -> Union[Dict[str, Dataset], None]:
        """Load from multiple data sources."""
        path_list = [path.strip() for path in mixed_paths.split(",")]
        datasets = {}

        hf_counter = 0
        for path in path_list:
            try:
                dataset = None
                base_name = self._extract_dataset_name(path)

                # Resolve S3 URIs first
                if path.startswith("s3://"):
                    path = self._resolve_s3_to_local(path)

                if self._is_huggingface_dataset(path):
                    # Get subset for this HF dataset
                    subset = self._get_hf_subset(hf_counter)
                    logger.info(f"Loading HF dataset: {path} with subset: {subset}")

                    temp_loader = DataLoader(
                        hf_subset=subset,
                        train_sample_size=self.train_sample_size,
                        eval_sample_size=self.eval_sample_size,
                        test_sample_size=self.test_sample_size
                    )
                    dataset = temp_loader._load_from_hub(path, split_type)
                    hf_counter += 1

                elif os.path.isdir(path):
                    dataset = self._load_from_directory(path, split_type)

                elif os.path.isfile(path):
                    if split_type != "train":
                        continue
                    dataset = self._load_file(path, split_type)

                if dataset is not None:
                    datasets[base_name] = dataset
                    logger.info(f"Loaded dataset: {base_name}")

            except Exception as e:
                logger.error(f"Failed to load: {path}, error: {e}")

        if not datasets and split_type == "train":
            raise ValueError("Failed to load any training data from mixed sources")

        return datasets if datasets else None

    def _extract_dataset_name(self, path: str) -> str:
        """Extract base name from dataset path."""
        if path.startswith("s3://"):
            return path.split("/")[-1] or path
        if self._is_huggingface_dataset(path):
            return path
        elif os.path.isdir(path):
            return os.path.basename(path.rstrip(os.path.sep))
        elif os.path.isfile(path):
            filename = os.path.basename(path)
            for ext in ['.json', '.jsonl', '.txt', '.csv', '.parquet']:
                if filename.endswith(ext):
                    return filename[:-len(ext)]
            return filename
        return path

    def _get_hf_subset(self, index: int) -> Optional[str]:
        """Get HF subset config for dataset at index."""
        if not self.hf_subsets:
            return self.DEFAULT_HF_SUBSET
        if index < len(self.hf_subsets):
            return self.hf_subsets[index]
        return self.DEFAULT_HF_SUBSET

    def _apply_sample_size(
        self,
        dataset: Dataset,
        split_type: str
    ) -> Optional[Dataset]:
        """Apply sample size limit."""
        size_map = {
            "train": self.train_sample_size,
            "eval": self.eval_sample_size,
            "dev": self.eval_sample_size,
            "test": self.test_sample_size
        }
        sample_size = size_map.get(split_type, -1)

        if sample_size == 0:
            logger.info(f"Sample size 0 for {split_type}, skipping dataset")
            return None
        elif sample_size == -1:
            return dataset

        original_size = len(dataset)
        if original_size > sample_size:
            dataset = dataset.select(range(sample_size))
            logger.info(f"Dataset ({split_type}): {original_size} -> {sample_size}")
        else:
            logger.info(f"Dataset ({split_type}): {original_size} (no limit needed)")

        return dataset

    def get_target_column(self, data: Union[Dataset, Dict[str, Dataset]]) -> str:
        """Get target column name from dataset."""
        if isinstance(data, dict):
            first_dataset = next(iter(data.values()))
            return self._get_single_target_column(first_dataset)
        return self._get_single_target_column(data)

    def _get_single_target_column(self, dataset: Dataset) -> str:
        """Get target column name from single dataset."""
        column_names = dataset.column_names

        # Three-column format: use third column
        if len(column_names) == 3:
            return column_names[2]

        # Standard column names
        if "score" in column_names:
            return "score"
        elif "label" in column_names:
            return "label"

        raise ValueError(f"Cannot determine target column. Columns: {column_names}")
