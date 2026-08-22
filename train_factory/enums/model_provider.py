"""Model provider and external model type enumerations."""

from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers."""

    # === Cloud Providers ===
    OPENAI = "openai"                   # OpenAI API
    AZURE = "azure"                     # Azure OpenAI
    ANTHROPIC = "anthropic"             # Anthropic Claude
    GOOGLE = "google"                   # Google AI (Gemini)
    COHERE = "cohere"                   # Cohere
    MISTRAL = "mistral"                 # Mistral AI
    DEEPSEEK = "deepseek"               # DeepSeek
    ZHIPU = "zhipu"                     # 智谱 AI (GLM)
    BAICHUAN = "baichuan"               # 百川
    QWEN = "qwen"                       # 通义千问 (阿里)
    MINIMAX = "minimax"                 # MiniMax
    MOONSHOT = "moonshot"               # Moonshot AI (Kimi)

    # === Self-hosted / Local ===
    XINFERENCE = "xinference"           # Xinference
    OLLAMA = "ollama"                   # Ollama
    VLLM = "vllm"                       # vLLM
    TRITON = "triton"                   # NVIDIA Triton
    TGI = "tgi"                         # Text Generation Inference
    LOCALAI = "localai"                 # LocalAI

    # === Embedding Specific ===
    JINA = "jina"                       # Jina AI
    VOYAGE = "voyage"                   # Voyage AI

    # === Custom ===
    CUSTOM = "custom"                   # Custom OpenAI-compatible endpoint


class ExternalModelType(str, Enum):
    """External model types for API configuration."""

    LLM = "llm"                         # Large Language Model
    EMBEDDING = "embedding"             # Text Embedding Model
    RERANK = "rerank"                   # Reranking Model
    MULTIMODAL = "multimodal"           # Multimodal Model (vision, etc.)
    SPEECH = "speech"                   # Speech-to-Text / Text-to-Speech


class ModelConfigStatus(str, Enum):
    """Model configuration status."""

    ACTIVE = "active"                   # Available for use
    INACTIVE = "inactive"               # Disabled
    ERROR = "error"                     # Connection error
    PENDING = "pending"                 # Awaiting verification


class ValidationErrorType(str, Enum):
    """Validation error types for API configuration."""

    INVALID_INPUT = "invalid_input"           # 输入参数无效
    INVALID_URL = "invalid_url"               # URL 格式无效
    AUTH_ERROR = "auth_error"                 # 认证失败
    CONNECTION_ERROR = "connection_error"     # 连接失败
    TIMEOUT_ERROR = "timeout_error"           # 连接超时
    API_ERROR = "api_error"                   # API 返回错误
    MODEL_NOT_FOUND = "model_not_found"       # 模型不存在
    UNKNOWN_ERROR = "unknown_error"           # 未知错误
