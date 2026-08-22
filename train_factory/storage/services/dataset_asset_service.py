"""
Dataset asset service — CRUD for dataset_assets.

Each dataset may have multiple file assets (train/eval/test splits,
batch files, etc.) tracked in the dataset_assets table.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..database import get_engine
from ..entities.dataset_asset_entity import DatasetAssetDB

logger = logging.getLogger(__name__)


class DatasetAssetService:
    """Service for managing dataset assets."""

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            self.engine = get_engine()
        return self.engine

    def create_asset(
        self,
        dataset_id: str,
        storage_uri: str,
        version: int = 1,
        asset_type: str = "data",
        file_format: Optional[str] = None,
        compression: Optional[str] = None,
        row_count: Optional[int] = None,
        byte_size: Optional[int] = None,
        checksum: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new asset for a dataset."""
        asset = DatasetAssetDB(
            dataset_id=dataset_id,
            storage_uri=storage_uri,
            version=version,
            asset_type=asset_type,
            file_format=file_format,
            compression=compression,
            row_count=row_count,
            byte_size=byte_size,
            checksum=checksum,
        )
        with Session(self._get_engine()) as session:
            session.add(asset)
            session.commit()
            session.refresh(asset)
            logger.info(
                f"Created asset {asset.asset_id} for dataset {dataset_id} "
                f"(type={asset_type}, uri={storage_uri})"
            )
            return asset.to_dict()

    def list_assets(
        self,
        dataset_id: str,
        version: Optional[int] = None,
        asset_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List assets for a dataset, optionally filtered by version/type."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetAssetDB).where(
                DatasetAssetDB.dataset_id == dataset_id
            )
            if version is not None:
                stmt = stmt.where(DatasetAssetDB.version == version)
            if asset_type is not None:
                stmt = stmt.where(DatasetAssetDB.asset_type == asset_type)
            stmt = stmt.order_by(DatasetAssetDB.created_at)
            assets = session.exec(stmt).all()
            return [a.to_dict() for a in assets]

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get a single asset by ID."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetAssetDB).where(
                DatasetAssetDB.asset_id == asset_id
            )
            asset = session.exec(stmt).first()
            return asset.to_dict() if asset else None

    def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset record. Returns True if found and deleted."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetAssetDB).where(
                DatasetAssetDB.asset_id == asset_id
            )
            asset = session.exec(stmt).first()
            if not asset:
                return False
            session.delete(asset)
            session.commit()
            logger.info(f"Deleted asset {asset_id}")
            return True

    def delete_assets_for_dataset(self, dataset_id: str) -> int:
        """Delete all assets belonging to a dataset. Returns count deleted."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetAssetDB).where(
                DatasetAssetDB.dataset_id == dataset_id
            )
            assets = session.exec(stmt).all()
            count = len(assets)
            for asset in assets:
                session.delete(asset)
            session.commit()
            if count:
                logger.info(f"Deleted {count} assets for dataset {dataset_id}")
            return count


dataset_asset_service = DatasetAssetService()
