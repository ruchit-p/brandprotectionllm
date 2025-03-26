import os
from celery import Celery
from app.config import get_settings
from app.beat_schedule import CELERYBEAT_SCHEDULE

settings = get_settings()

# Initialize Celery
celery_app = Celery(
    'brand_protection_worker',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.rekognition_tasks',
        'app.tasks.analysis_tasks',
        'app.tasks.scheduled_tasks',
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 2,  # 2 hours max runtime
    task_soft_time_limit=3600,  # 1 hour soft limit
    worker_max_tasks_per_child=200,
    broker_connection_retry_on_startup=True,
    
    # Result expiration (1 day by default)
    result_expires=86400,
    
    # Add beat schedule
    beat_schedule=CELERYBEAT_SCHEDULE,
    
    # Store the schedule in Redis to make it persistent
    beat_scheduler='redbeat.RedBeatScheduler',
    redbeat_redis_url=settings.CELERY_BROKER_URL,
    
    # Task always eager for testing
    task_always_eager=os.environ.get('CELERY_TASK_ALWAYS_EAGER', 'False').lower() == 'true',
)

# Configure task routes for different queues
celery_app.conf.task_routes = {
    'app.tasks.rekognition_tasks.*': {'queue': 'rekognition'},
    'app.tasks.analysis_tasks.*': {'queue': 'analysis'},
    'app.tasks.scheduled_tasks.*': {'queue': 'maintenance'},
}

# Configure task retry policy
celery_app.conf.task_default_retry_delay = 60  # 1 minute
celery_app.conf.task_max_retries = 3

# Set concurrency to control maximum number of worker processes
celery_app.conf.worker_concurrency = os.cpu_count() or 4

if __name__ == '__main__':
    celery_app.start() 