#!/bin/bash

# Set Python path
export PYTHONPATH=$(pwd)

# Load environment variables if .env file exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Set test environment variables
export CELERY_TASK_ALWAYS_EAGER=True
export DB_NAME=brand_protection_test

# Run tests with pytest
pytest -xvs "$@" 