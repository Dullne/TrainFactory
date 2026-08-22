"""Database entities for TrainFactory."""

from .training_task_entity import TrainingTaskDB
from .model_registry_entity import ModelRegistryDB, ModelVersionDB
from .deployment_entity import DeploymentDB
from .loaded_adapter_entity import LoadedAdapterDB
from .model_config_entity import ModelConfigDB
from .dataset_entity import DatasetDB
from .user_entity import UserDB
from .audit_log_entity import AuditLogDB, AuditAction, AuditResource
from .evaluation_task_entity import (
    EvaluationTaskDB,
    EvaluationFramework,
    EvaluationStatus,
    # Backward compatibility aliases
    DeepEvaluationTaskDB,
    DeepEvaluationStatus,
)
from .generation_task_entity import GenerationTaskDB, GenerationStatus, GenerationMode
from .training_task_event_entity import TrainingTaskEventDB
from .milvus_collection_entity import MilvusCollectionDB, CollectionDatasetLinkDB
from .external_sync_entity import (
    ExternalSyncTaskDB, ExternalSyncBatchDB,
    ExternalSyncGenerationDB, ExternalSyncTrainingDB,
)
from .external_api_config_entity import ExternalApiConfigDB
from .dataset_asset_entity import DatasetAssetDB
from .dataset_lineage_entity import DatasetLineageEdgeDB
from ...enums.sync_status import (
    SyncStatus, BatchStatus, SyncGenerationStatus, SyncTrainingStatus,
)

__all__ = [
    "TrainingTaskDB",
    "ModelRegistryDB",
    "ModelVersionDB",
    "DeploymentDB",
    "LoadedAdapterDB",
    "ModelConfigDB",
    "DatasetDB",
    "UserDB",
    "AuditLogDB",
    "AuditAction",
    "AuditResource",
    # Evaluation (unified)
    "EvaluationTaskDB",
    "EvaluationFramework",
    "EvaluationStatus",
    # Backward compatibility
    "DeepEvaluationTaskDB",
    "DeepEvaluationStatus",
    # Generation
    "GenerationTaskDB",
    "GenerationStatus",
    "GenerationMode",
    "TrainingTaskEventDB",
    # Milvus collection registry
    "MilvusCollectionDB",
    "CollectionDatasetLinkDB",
    # External sync
    "ExternalSyncTaskDB",
    "ExternalSyncBatchDB",
    "ExternalSyncGenerationDB",
    "ExternalSyncTrainingDB",
    "SyncStatus",
    "BatchStatus",
    "SyncGenerationStatus",
    "SyncTrainingStatus",
    # External API config
    "ExternalApiConfigDB",
    # Dataset storage refactor
    "DatasetAssetDB",
    "DatasetLineageEdgeDB",
]
