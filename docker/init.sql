-- TrainFactory Database Initialization Script
-- This script creates the necessary tables for the training task management
--
-- 注意：本脚本只覆盖核心业务表。其余表/列（同步、Milvus 集合、token_version
-- 等）由 Alembic 迁移（train_factory/storage/migrations，038-047）在 API 启动
-- 时自动创建（db_auto_migrate=True）。禁止在 db_auto_migrate=False 下部署。

-- Set character set
SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- Use the database
USE train_factory;

-- ============================================================
-- Training Tasks Table
-- ============================================================
CREATE TABLE IF NOT EXISTS training_tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL UNIQUE,
    task_name VARCHAR(255),
    description TEXT,

    -- Model & Training Type
    model_type VARCHAR(32) DEFAULT 'embedding',
    training_method VARCHAR(32) DEFAULT 'sft',
    model_architecture VARCHAR(32) DEFAULT 'encoder',

    -- User isolation
    user_id VARCHAR(64),

    -- Model info
    base_model_path VARCHAR(1024),
    final_model_path VARCHAR(1024),

    -- Dataset info
    train_dataset_path VARCHAR(1024),

    -- Output & Device
    output_dir VARCHAR(1024),
    embedding_dim INT,
    device VARCHAR(64) DEFAULT 'cuda:0',

    -- LoRA & Checkpoints
    is_lora BOOLEAN DEFAULT FALSE,
    checkpoints JSON,
    best_checkpoint_path VARCHAR(1024),
    loss_data TEXT,

    -- Model Registry Link
    trained_model_registry_id VARCHAR(36),

    -- RL & Two-Stage Training (New)
    rl_config JSON,
    loss_config JSON,
    parent_task_id VARCHAR(36),
    sft_checkpoint_path VARCHAR(1024),

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,

    -- Training parameters (JSON)
    training_params JSON,

    -- Results (JSON)
    final_metrics JSON,

    -- Process info
    run_token VARCHAR(36),
    process_pid INT,
    process_status VARCHAR(50),
    process_create_time DOUBLE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,

    -- Indexes
    INDEX idx_task_id (task_id),
    INDEX idx_status (status),
    INDEX idx_model_type (model_type),
    INDEX idx_training_method (training_method),
    INDEX idx_user_id (user_id),
    INDEX idx_trained_model_registry_id (trained_model_registry_id),
    INDEX idx_parent_task_id (parent_task_id),
    INDEX idx_training_run_token (run_token),
    INDEX idx_created_at (created_at),
    INDEX idx_task_user_created (user_id, created_at),
    INDEX idx_task_type_status (model_type, status),
    UNIQUE KEY uq_task_user_name (user_id, task_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Model Registry Table
-- ============================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_id VARCHAR(36) NOT NULL UNIQUE,
    model_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(256),
    version VARCHAR(50) DEFAULT 'v1.0.0',
    model_type VARCHAR(50) NOT NULL,

    -- Source info
    source_task_id VARCHAR(36),
    source_model_id VARCHAR(36),
    base_model_path VARCHAR(1024),
    model_path VARCHAR(1024) NOT NULL,
    model_path_hash VARCHAR(64),
    best_checkpoint_path VARCHAR(1024),
    checkpoints JSON,

    -- Model Properties
    embedding_dim INT,
    source_type VARCHAR(32) DEFAULT 'trained',
    is_adapter BOOLEAN DEFAULT FALSE,

    -- Download Info (for downloaded models)
    download_source VARCHAR(32),
    remote_repo VARCHAR(512),
    download_status VARCHAR(32),
    download_progress INT DEFAULT 0,
    download_error TEXT,

    -- Metadata
    description TEXT,
    tags JSON,
    category VARCHAR(100),
    extra_metadata JSON,

    -- Performance metrics
    metrics JSON,
    file_size BIGINT,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'registered',
    is_latest BOOLEAN DEFAULT TRUE,

    -- User isolation
    user_id VARCHAR(64),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_model_id (model_id),
    INDEX idx_model_name (model_name),
    INDEX idx_model_type (model_type),
    INDEX idx_source_model_id (source_model_id),
    INDEX idx_source_type (source_type),
    INDEX idx_download_status (download_status),
    INDEX idx_model_path_hash (model_path_hash),
    INDEX idx_status (status),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_model_user_created (user_id, created_at),
    INDEX idx_model_type_status (model_type, status),
    INDEX ix_model_registry_source_task_id (source_task_id),
    UNIQUE KEY uq_model_user_name_version (user_id, model_name, version),
    UNIQUE KEY uq_model_path_hash (model_path_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Model Versions Table
-- ============================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    version_id VARCHAR(36) NOT NULL UNIQUE,
    model_id VARCHAR(36) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_path VARCHAR(1024) NOT NULL,

    -- Changelog
    changelog TEXT,

    -- Metrics
    metrics JSON,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_version_id (version_id),
    INDEX idx_model_id (model_id),
    INDEX idx_created_at (created_at),
    INDEX idx_version_model_created (model_id, created_at),
    UNIQUE KEY uq_version_model_version (model_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Deployments Table
-- ============================================================
CREATE TABLE IF NOT EXISTS deployments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    deployment_id VARCHAR(36) NOT NULL UNIQUE,
    model_id VARCHAR(36) NOT NULL,
    model_uid VARCHAR(255),

    -- Deployment name
    deployment_name VARCHAR(255),

    -- Configuration
    xinference_endpoint VARCHAR(512) NOT NULL,
    replica INT DEFAULT 1,
    gpu_memory_utilization FLOAT DEFAULT 0.9,
    config JSON,

    -- Container Mode
    deploy_mode VARCHAR(32) DEFAULT 'shared',
    container_name VARCHAR(255),
    gpu_id INT,
    port INT,

    -- Inference Framework
    inference_framework VARCHAR(32) DEFAULT 'xinference',

    -- LoRA Hot-Loading
    enable_lora BOOLEAN DEFAULT FALSE,
    max_loras INT DEFAULT 4,
    max_lora_rank INT DEFAULT 64,

    -- External API deployment link
    external_api_config_id VARCHAR(36),

    -- Health Check
    health_check_url VARCHAR(512),
    last_health_check TIMESTAMP NULL,
    health_status VARCHAR(32) DEFAULT 'UNKNOWN',

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,

    -- User isolation
    user_id VARCHAR(64),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    stopped_at TIMESTAMP NULL,

    -- Indexes
    INDEX idx_deployment_id (deployment_id),
    INDEX idx_model_id (model_id),
    INDEX idx_status (status),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_deploy_mode (deploy_mode),
    INDEX idx_inference_framework (inference_framework),
    INDEX idx_deployment_model_status (model_id, status),
    INDEX idx_deployment_user_created (user_id, created_at),
    INDEX idx_deployment_external_api_config_id (external_api_config_id),
    INDEX idx_deployment_external_api_status (external_api_config_id, status),
    UNIQUE KEY uq_deployment_user_name (user_id, deployment_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- External Model Configurations Table
-- ============================================================
CREATE TABLE IF NOT EXISTS model_configs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_id VARCHAR(36) NOT NULL UNIQUE,
    config_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(256),

    -- Source type: external_api, local_deployed
    source_type VARCHAR(32) DEFAULT 'external_api',

    -- Local Deployment Link (for source_type=local_deployed)
    registry_id VARCHAR(36),
    deployment_id VARCHAR(36),
    container_name VARCHAR(255),
    inference_framework VARCHAR(32),

    -- Model type: llm, embedding, rerank, multimodal, speech
    model_type VARCHAR(50) NOT NULL,

    -- Provider: openai, azure, xinference, ollama, custom, etc.
    provider VARCHAR(50) NOT NULL,

    -- Connection settings
    api_endpoint VARCHAR(1024) NOT NULL,
    api_key VARCHAR(512),
    model_name VARCHAR(255) NOT NULL,

    -- Model Properties
    embedding_dim INT DEFAULT 1024,
    max_input_length INT,

    -- Provider-specific config (JSON)
    provider_config JSON,

    -- Default parameters for API calls (JSON)
    default_params JSON,

    -- Metadata
    description TEXT,
    tags JSON,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    is_default BOOLEAN DEFAULT FALSE,

    -- Health check
    last_check_status VARCHAR(50),
    last_check_time TIMESTAMP NULL,
    last_check_error TEXT,

    -- User isolation
    user_id VARCHAR(64),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_config_id (config_id),
    INDEX idx_config_name (config_name),
    INDEX idx_source_type (source_type),
    INDEX idx_registry_id (registry_id),
    INDEX idx_deployment_id (deployment_id),
    INDEX idx_model_type (model_type),
    INDEX idx_provider (provider),
    INDEX idx_status (status),
    INDEX idx_user_id (user_id),
    INDEX idx_is_default (is_default),
    INDEX idx_config_user_type (user_id, model_type),
    INDEX idx_config_provider_status (provider, status),
    UNIQUE KEY uq_config_user_name (user_id, config_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Datasets Table
-- ============================================================
CREATE TABLE IF NOT EXISTS datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id VARCHAR(36) NOT NULL UNIQUE,
    dataset_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(256),
    description TEXT,

    -- Dataset Type
    -- embedding_pair, embedding_triplet, rerank_pair, rerank_listwise,
    -- sft_instruct, dpo_preference, rl_reward, custom
    dataset_type VARCHAR(50) NOT NULL DEFAULT 'custom',

    -- Usage: train (训练), eval (评估)
    `usage` VARCHAR(20) NOT NULL DEFAULT 'train',

    -- 适用模型标签（可选，用户自行标注，可多选）: embedding, rerank, llm
    model_type JSON DEFAULT NULL,

    -- Source Info
    -- source_type: uploaded, huggingface, modelscope, local
    source_type VARCHAR(32) DEFAULT 'uploaded',
    source_path VARCHAR(1024),
    source_dataset_id VARCHAR(36),
    remote_repo VARCHAR(512),
    hf_subset VARCHAR(128),

    -- Storage
    storage_path VARCHAR(1024),
    storage_path_hash VARCHAR(64),
    file_format VARCHAR(32) DEFAULT 'parquet',

    -- Object storage (MinIO/S3)
    storage_backend VARCHAR(16) DEFAULT 'local',
    storage_uri VARCHAR(2048),
    version INT DEFAULT 1,

    -- Schema Info
    columns JSON,
    sample_data JSON,
    content_schema JSON,

    -- Statistics
    num_rows INT,
    num_train INT,
    num_eval INT,
    num_test INT,
    file_size BIGINT,

    -- Provenance
    source_task_type VARCHAR(32),
    source_task_id VARCHAR(36),
    generation_run_token VARCHAR(36),

    -- Metadata
    tags JSON,
    extra_metadata JSON,

    -- Status
    -- registered, uploading, processing, downloading, ready, error, archived, deleting
    status VARCHAR(50) NOT NULL DEFAULT 'registered',
    deletion_owner VARCHAR(128),
    error_message TEXT,

    -- User isolation
    user_id VARCHAR(64),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_dataset_id (dataset_id),
    INDEX idx_dataset_name (dataset_name),
    INDEX idx_dataset_type (dataset_type),
    INDEX idx_usage (`usage`),
    INDEX idx_source_type (source_type),
    INDEX idx_storage_path_hash (storage_path_hash),
    INDEX idx_status (status),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_dataset_source_dataset (source_dataset_id),
    INDEX idx_dataset_user_created (user_id, created_at),
    INDEX idx_dataset_type_status (dataset_type, status),
    INDEX idx_dataset_generation_run_token (generation_run_token),
    UNIQUE KEY uq_dataset_user_name (user_id, dataset_name),
    UNIQUE KEY uq_dataset_storage_path_hash (storage_path_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Dataset Assets Table
-- ============================================================
CREATE TABLE IF NOT EXISTS dataset_assets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    asset_id VARCHAR(36) NOT NULL UNIQUE,
    dataset_id VARCHAR(36) NOT NULL,
    version INT DEFAULT 1,

    -- Asset type: data, split_train, split_eval, split_test
    asset_type VARCHAR(32) DEFAULT 'data',

    -- Storage URI (s3://bucket/key)
    storage_uri VARCHAR(2048) NOT NULL,

    -- File metadata
    file_format VARCHAR(32),
    compression VARCHAR(16),
    row_count INT,
    byte_size BIGINT,
    checksum VARCHAR(128),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_asset_id (asset_id),
    INDEX idx_asset_dataset_id (dataset_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Dataset Lineage Edges Table
-- ============================================================
CREATE TABLE IF NOT EXISTS dataset_lineage_edges (
    id INT AUTO_INCREMENT PRIMARY KEY,
    edge_id VARCHAR(36) NOT NULL UNIQUE,
    edge_key VARCHAR(64) NOT NULL,

    -- Source dataset (NULL for root edges like sync_fetched)
    from_dataset_id VARCHAR(36),
    -- Target dataset
    to_dataset_id VARCHAR(36) NOT NULL,

    -- Relation type: sync_fetched, merged, qa_extracted, qa_filtered,
    -- training_generated, deep_eval_generated, training_input
    relation_type VARCHAR(32) NOT NULL,

    -- Task that performed the transformation
    op_task_type VARCHAR(32),
    op_task_id VARCHAR(36),

    -- Snapshot of transformation parameters
    op_params JSON,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_edge_id (edge_id),
    UNIQUE KEY uq_lineage_edge_key (edge_key),
    INDEX idx_from_dataset_id (from_dataset_id),
    INDEX idx_to_dataset_id (to_dataset_id),
    INDEX idx_op_task_id (op_task_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Evaluation Tasks Table
-- ============================================================
CREATE TABLE IF NOT EXISTS evaluation_tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL UNIQUE,
    run_token VARCHAR(36),

    -- Basic info
    task_name VARCHAR(255),
    description TEXT,

    -- Evaluation framework and type
    eval_framework VARCHAR(32) NOT NULL DEFAULT 'mteb',
    eval_type VARCHAR(32) NOT NULL DEFAULT 'single',

    -- Configuration (JSON)
    model_configs JSON,
    dataset_configs JSON,
    field_mapping JSON,
    metrics JSON,
    llm_config JSON,
    worker_groups JSON,

    -- Evaluation parameters
    max_samples INT,
    batch_size INT NOT NULL DEFAULT 50,
    workers INT NOT NULL DEFAULT 8,
    model_workers INT NOT NULL DEFAULT 2,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress FLOAT NOT NULL DEFAULT 0.0,
    current_model VARCHAR(255),
    current_dataset VARCHAR(255),
    error_message TEXT,
    total_samples INT NOT NULL DEFAULT 0,
    processed_samples INT NOT NULL DEFAULT 0,

    -- Per-model progress (JSON)
    model_progress JSON,

    -- Results (JSON)
    results JSON,
    results_path VARCHAR(1024),

    -- User isolation
    user_id VARCHAR(64),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,

    -- Indexes
    INDEX idx_eval_task_id (task_id),
    INDEX idx_eval_status (status),
    INDEX idx_eval_user_created (user_id, created_at),
    INDEX idx_eval_framework (eval_framework),
    INDEX idx_eval_run_token (run_token)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Show table structures
-- ============================================================
-- ============ 认证与审计表（与迁移 038_add_auth_and_audit_tables 对齐）============
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL UNIQUE,
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(256) UNIQUE,
    hashed_password VARCHAR(256) NOT NULL,
    is_active TINYINT(1) NOT NULL,
    is_admin TINYINT(1) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    KEY ix_users_is_active (is_active)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    log_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    username VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(36),
    method VARCHAR(10) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    request_body JSON,
    status_code INT NOT NULL,
    response_summary VARCHAR(1000),
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    extra_data JSON,
    created_at DATETIME NOT NULL,
    KEY idx_audit_action (action),
    KEY idx_audit_resource (resource_type, resource_id),
    KEY ix_audit_logs_log_id (log_id),
    KEY ix_audit_logs_user_id (user_id)
);

DESCRIBE training_tasks;
DESCRIBE model_registry;
DESCRIBE model_versions;
DESCRIBE deployments;
DESCRIBE model_configs;
DESCRIBE datasets;
DESCRIBE dataset_assets;
DESCRIBE dataset_lineage_edges;
DESCRIBE evaluation_tasks;
DESCRIBE users;
DESCRIBE audit_logs;

-- Confirm initialization
SELECT 'TrainFactory database initialized successfully!' AS message;
