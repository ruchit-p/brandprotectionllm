#!/usr/bin/env python3
"""
Database migration runner for Brand Protection system
"""
import os
import sys
import glob
import logging
import argparse
import psycopg2
from psycopg2.errors import DuplicateTable, UndefinedTable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('migrations')

def get_db_connection(host, port, dbname, user, password):
    """Create a database connection"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def create_version_table(conn):
    """Create the schema_versions table if it doesn't exist"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_versions (
            id SERIAL PRIMARY KEY,
            version VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        """)
        conn.commit()
        logger.info("Schema version table ready")
    except Exception as e:
        logger.error(f"Error creating version table: {e}")
        conn.rollback()
        sys.exit(1)

def get_applied_migrations(conn):
    """Get list of already applied migrations"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_versions ORDER BY id")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        if isinstance(e, UndefinedTable):
            logger.info("Schema version table doesn't exist yet")
            return []
        logger.error(f"Error getting applied migrations: {e}")
        return []

def run_migration(conn, migration_file, dry_run=False):
    """Run a single migration file"""
    try:
        migration_name = os.path.basename(migration_file)
        
        with open(migration_file, 'r') as f:
            sql = f.read()
            
        if dry_run:
            logger.info(f"Would run migration: {migration_name}")
            return True
            
        cursor = conn.cursor()
        logger.info(f"Running migration: {migration_name}")
        cursor.execute(sql)
        conn.commit()
        
        logger.info(f"Migration completed: {migration_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error running migration {migration_file}: {e}")
        conn.rollback()
        return False

def main():
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument('--host', default=os.environ.get('DB_HOST', 'localhost'), help='Database host')
    parser.add_argument('--port', type=int, default=int(os.environ.get('DB_PORT', '5432')), help='Database port')
    parser.add_argument('--dbname', default=os.environ.get('DB_NAME', 'brand_protection'), help='Database name')
    parser.add_argument('--user', default=os.environ.get('DB_USER', 'postgres'), help='Database user')
    parser.add_argument('--password', default=os.environ.get('DB_PASSWORD', 'postgres'), help='Database password')
    parser.add_argument('--dir', default='migrations', help='Migrations directory')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actually executing migrations')
    
    args = parser.parse_args()
    
    # Get full path to migrations directory
    migrations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dir)
    if not os.path.exists(migrations_dir):
        logger.error(f"Migrations directory {migrations_dir} does not exist")
        sys.exit(1)
    
    # Connect to database
    conn = get_db_connection(args.host, args.port, args.dbname, args.user, args.password)
    
    try:
        # Create version table if needed
        create_version_table(conn)
        
        # Get list of applied migrations
        applied = get_applied_migrations(conn)
        
        # Get all migration files
        migration_files = sorted(glob.glob(os.path.join(migrations_dir, '*.sql')))
        
        if not migration_files:
            logger.info(f"No migration files found in {migrations_dir}")
            return
        
        # Run migrations that haven't been applied
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for migration_file in migration_files:
            migration_name = os.path.basename(migration_file).split('.')[0]
            
            if migration_name in applied:
                logger.info(f"Skipping already applied migration: {migration_name}")
                skipped_count += 1
                continue
                
            if run_migration(conn, migration_file, args.dry_run):
                success_count += 1
            else:
                failed_count += 1
                
        logger.info(f"Migration summary: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
        
    finally:
        conn.close()

if __name__ == '__main__':
    main() 