"""Training result container."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingResult:
    """
    Training result container.

    Attributes:
        model: Trained model object
        save_dir: Directory where the model was saved
        final_metrics: Final evaluation metrics (e.g., accuracy, f1)
        training_stats: Training statistics (e.g., loss history)
        best_checkpoint_path: Path to the best checkpoint
        task_id: Associated training task ID (for API tracking)
    """
    model: Any
    save_dir: str
    final_metrics: Optional[Dict[str, float]] = None
    training_stats: Optional[Dict[str, Any]] = None
    best_checkpoint_path: Optional[str] = None
    task_id: Optional[str] = None
