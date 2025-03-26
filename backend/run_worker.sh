#!/bin/bash

# Set Python path
export PYTHONPATH=$(pwd)

# Load environment variables
set -a
source .env
set +a

# Check which queue to run
if [ "$1" = "rekognition" ]; then
    echo "Starting Rekognition worker..."
    celery -A app.celery_worker worker -Q rekognition -l info -n rekognition@%h
elif [ "$1" = "analysis" ]; then
    echo "Starting Analysis worker..."
    celery -A app.celery_worker worker -Q analysis -l info -n analysis@%h
elif [ "$1" = "flower" ]; then
    echo "Starting Flower monitoring..."
    celery -A app.celery_worker flower --port=5555
else
    # Default to running all queues
    echo "Starting worker with all queues..."
    celery -A app.celery_worker worker -l info
fi 