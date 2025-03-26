from celery.schedules import crontab

# Define periodic task schedules
CELERYBEAT_SCHEDULE = {
    # Check model status every 5 minutes
    'check-model-status-every-5-minutes': {
        'task': 'app.tasks.scheduled_tasks.check_all_model_statuses',
        'schedule': 300.0,  # Every 5 minutes
        'options': {
            'queue': 'rekognition',
        }
    },
    
    # Clean up old task results daily at midnight
    'cleanup-task-results-daily': {
        'task': 'app.tasks.scheduled_tasks.cleanup_old_task_results',
        'schedule': crontab(hour=0, minute=0),  # Midnight every day
        'options': {
            'queue': 'maintenance',
        }
    },
    
    # Check for interrupted tasks on startup and every hour
    'recover-interrupted-tasks': {
        'task': 'app.tasks.scheduled_tasks.recover_interrupted_tasks',
        'schedule': crontab(minute=0),  # Every hour
        'options': {
            'queue': 'maintenance',
        }
    },
    
    # Update database statistics daily for query optimization
    'update-database-stats': {
        'task': 'app.tasks.scheduled_tasks.update_database_stats',
        'schedule': crontab(hour=1, minute=30),  # 1:30 AM every day
        'options': {
            'queue': 'maintenance',
        }
    },
    
    # Check AWS credentials permissions daily
    'verify-aws-permissions': {
        'task': 'app.tasks.scheduled_tasks.verify_aws_permissions',
        'schedule': crontab(hour=2, minute=0),  # 2:00 AM every day
        'options': {
            'queue': 'maintenance',
        }
    }
} 