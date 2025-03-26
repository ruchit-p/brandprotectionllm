-- Migration script to add task tracking and analysis status columns
-- For existing database deployments

-- First create a version tracking table if it doesn't exist
CREATE TABLE IF NOT EXISTS schema_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Check if this migration has already been applied
DO $$
DECLARE
    migration_exists BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM schema_versions WHERE version = '01_add_task_tracking'
    ) INTO migration_exists;
    
    IF migration_exists THEN
        RAISE NOTICE 'Migration 01_add_task_tracking already applied, skipping';
    ELSE
        -- Add task_id column to brand_custom_models if it doesn't exist
        ALTER TABLE brand_custom_models 
        ADD COLUMN IF NOT EXISTS task_id VARCHAR(255);
        
        -- Add analysis status columns to websites if they don't exist
        ALTER TABLE websites 
        ADD COLUMN IF NOT EXISTS analysis_status VARCHAR(50) DEFAULT 'PENDING';
        
        ALTER TABLE websites 
        ADD COLUMN IF NOT EXISTS analysis_status_message TEXT;
        
        ALTER TABLE websites 
        ADD COLUMN IF NOT EXISTS analysis_completed_at TIMESTAMP;
        
        -- Add indexes for performance
        CREATE INDEX IF NOT EXISTS idx_brand_custom_models_task_id 
        ON brand_custom_models(task_id);
        
        CREATE INDEX IF NOT EXISTS idx_websites_analysis_status 
        ON websites(analysis_status);
        
        -- Record the migration
        INSERT INTO schema_versions (version, description)
        VALUES ('01_add_task_tracking', 'Add task tracking columns and analysis status');
        
        RAISE NOTICE 'Migration 01_add_task_tracking applied successfully';
    END IF;
END $$; 