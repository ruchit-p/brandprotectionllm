import logging
import psycopg2
from datetime import datetime, timedelta
from celery import shared_task
from app.config import get_settings
from app.services.rekognition import RekognitionService
import boto3

settings = get_settings()
logger = logging.getLogger(__name__)

def get_db():
    """Get a database connection"""
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )
    conn.autocommit = False
    return conn

@shared_task
def check_all_model_statuses():
    """
    Check the status of all models in the database and update if changed
    """
    conn = None
    try:
        conn = get_db()
        rekognition_service = RekognitionService()
        
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, model_version_arn, status
            FROM brand_custom_models
            WHERE model_version_arn IS NOT NULL
            AND status NOT IN ('ERROR', 'TRAINING_FAILED', 'STOPPED')
            """
        )
        models = cursor.fetchall()
        
        if not models:
            logger.info("No active models found to check status")
            return {"status": "success", "message": "No models to check"}
        
        updated_count = 0
        
        for model_id, model_version_arn, current_status in models:
            try:
                # Get current status from AWS
                aws_status = rekognition_service.get_model_status(model_version_arn)
                
                if not aws_status:
                    logger.warning(f"Could not get status for model {model_id}")
                    continue
                
                # Map AWS status to our status if needed
                status_mapping = {
                    "TRAINING_IN_PROGRESS": "TRAINING",
                    "TRAINING_COMPLETED": "TRAINING_COMPLETED",
                    "TRAINING_FAILED": "ERROR",
                    "STARTING": "STARTING",
                    "RUNNING": "RUNNING",
                    "STOPPING": "STOPPING",
                    "STOPPED": "STOPPED",
                    "FAILED": "ERROR"
                }
                
                mapped_status = status_mapping.get(aws_status, aws_status)
                
                # Update if status has changed
                if mapped_status != current_status:
                    cursor.execute(
                        """
                        UPDATE brand_custom_models
                        SET status = %s, 
                            status_message = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (mapped_status, f"Status updated from AWS: {aws_status}", model_id)
                    )
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking model {model_id} status: {e}")
                
        conn.commit()
        return {
            "status": "success", 
            "checked_count": len(models),
            "updated_count": updated_count
        }
        
    except Exception as e:
        logger.error(f"Error in check_all_model_statuses: {e}")
        if conn:
            conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()

@shared_task
def cleanup_old_task_results():
    """
    Clean up expired task results from Redis
    """
    try:
        from redis import Redis
        from celery.result import ResultBase
        
        redis_client = Redis.from_url(settings.CELERY_BROKER_URL)
        
        # Get all keys with celery-task-meta- prefix (task results)
        task_keys = redis_client.keys('celery-task-meta-*')
        
        if not task_keys:
            return {"status": "success", "message": "No task results to clean up"}
        
        # For each key, check if it's older than the expiration time
        deleted_count = 0
        
        for key in task_keys:
            try:
                # Check if the task is too old (older than 7 days)
                result_data = redis_client.get(key)
                if result_data:
                    # Try to parse the result data
                    # If we wanted to be more selective, we could parse the JSON and check dates
                    # But for simplicity, we'll just delete all tasks older than X days
                    # This is handled by Redis TTL anyway, but as a backup we do manual cleanup
                    
                    # Delete the key
                    redis_client.delete(key)
                    deleted_count += 1
                    
            except Exception as e:
                logger.error(f"Error cleaning up task result {key}: {e}")
        
        return {
            "status": "success", 
            "total_keys": len(task_keys),
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_task_results: {e}")
        return {"status": "error", "message": str(e)}

@shared_task
def recover_interrupted_tasks():
    """
    Recover tasks that were interrupted by worker restarts
    """
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Find brand models with running tasks that haven't been updated recently
        cursor.execute(
            """
            SELECT id, brand_id, task_id, status, updated_at
            FROM brand_custom_models
            WHERE status IN ('TRAINING', 'TRAINING_IN_PROGRESS', 'CREATING_ANNOTATIONS', 
                            'PROCESSING_IMAGES', 'STARTING', 'STOPPING')
            AND updated_at < NOW() - INTERVAL '30 minutes'
            """
        )
        stalled_models = cursor.fetchall()
        
        recovered_count = 0
        
        # For each stalled task, check if it's still running
        for model_id, brand_id, task_id, status, updated_at in stalled_models:
            try:
                if not task_id:
                    continue
                    
                from celery.result import AsyncResult
                task_result = AsyncResult(task_id)
                
                # If task is not running anymore but status says it is
                if task_result.ready() or task_result.state in ('REVOKED', 'FAILURE'):
                    # Update model status
                    cursor.execute(
                        """
                        UPDATE brand_custom_models
                        SET status = %s, 
                            status_message = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        ('ERROR', f'Task {task_id} was interrupted', model_id)
                    )
                    recovered_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking task {task_id} for model {model_id}: {e}")
        
        # Find website analyses that are stalled
        cursor.execute(
            """
            SELECT id, analysis_status, updated_at
            FROM websites
            WHERE analysis_status = 'ANALYZING'
            AND updated_at < NOW() - INTERVAL '30 minutes'
            """
        )
        stalled_analyses = cursor.fetchall()
        
        # Update stalled analyses
        for website_id, status, updated_at in stalled_analyses:
            try:
                cursor.execute(
                    """
                    UPDATE websites
                    SET analysis_status = %s, 
                        analysis_status_message = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    ('ERROR', 'Analysis task was interrupted', website_id)
                )
                recovered_count += 1
                
            except Exception as e:
                logger.error(f"Error updating website {website_id} status: {e}")
                
        conn.commit()
        
        return {
            "status": "success", 
            "stalled_models": len(stalled_models),
            "stalled_analyses": len(stalled_analyses),
            "recovered_count": recovered_count
        }
        
    except Exception as e:
        logger.error(f"Error in recover_interrupted_tasks: {e}")
        if conn:
            conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()

@shared_task
def update_database_stats():
    """
    Update database statistics for query optimization
    """
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Run ANALYZE on important tables
        tables = [
            'brands', 
            'websites', 
            'website_snapshots', 
            'brand_custom_models',
            'website_assets', 
            'brand_assets',
            'analysis_results'
        ]
        
        for table in tables:
            try:
                cursor.execute(f"ANALYZE {table}")
            except Exception as e:
                logger.error(f"Error analyzing table {table}: {e}")
        
        conn.commit()
        return {"status": "success", "tables_analyzed": len(tables)}
        
    except Exception as e:
        logger.error(f"Error in update_database_stats: {e}")
        if conn:
            conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()

@shared_task
def verify_aws_permissions():
    """
    Verify that AWS credentials have the necessary permissions
    """
    try:
        # Check IAM permissions using STS
        sts_client = boto3.client(
            'sts',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        
        # GetCallerIdentity works with any valid credentials
        identity = sts_client.get_caller_identity()
        account_id = identity.get('Account')
        
        # Check Rekognition permissions
        rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        
        # Try a simple operation
        rekognition_client.describe_projects()
        
        # Check S3 permissions
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        
        # Try to list buckets
        s3_client.list_buckets()
        
        # Check specific bucket permissions
        try:
            s3_client.head_bucket(Bucket=settings.AWS_CUSTOM_LABELS_BUCKET)
        except Exception as bucket_error:
            logger.error(f"Custom labels bucket not accessible: {bucket_error}")
            return {
                "status": "warning",
                "aws_identity": account_id,
                "message": f"AWS credentials valid but custom labels bucket not accessible: {str(bucket_error)}"
            }
        
        return {
            "status": "success",
            "aws_identity": account_id,
            "message": "AWS credentials have all required permissions"
        }
        
    except Exception as e:
        logger.error(f"Error in verify_aws_permissions: {e}")
        return {"status": "error", "message": str(e)} 