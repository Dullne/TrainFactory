"""
Deployment module for TrainFactory.

Provides Xinference integration for model deployment.
"""

from .xinference_client import XinferenceClient
from .deployment_service import DeploymentService, deployment_service

__all__ = [
    "XinferenceClient",
    "DeploymentService",
    "deployment_service",
]
