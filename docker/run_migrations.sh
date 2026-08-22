#!/bin/bash
# Run Alembic database migrations
#
# This script runs all pending database migrations.
# It should be executed after the database is ready.
#
# Usage:
#   ./run_migrations.sh              # Run migrations
#   ./run_migrations.sh --dry-run    # Show pending migrations without applying

set -e

# Change to the migrations directory
cd /app/train_factory/storage/migrations

if [ "$1" = "--dry-run" ]; then
    echo "=== Pending migrations ==="
    alembic history --indicate-current
    echo ""
    echo "=== Current head ==="
    alembic heads
else
    echo "=== Running database migrations ==="
    alembic upgrade head
    echo "=== Migrations completed ==="
fi
