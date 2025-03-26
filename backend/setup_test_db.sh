#!/bin/bash

# Set the database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
TEST_DB_NAME="brand_protection_test"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up test database for Brand Protection...${NC}"

# Check if PostgreSQL is running
if ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER > /dev/null 2>&1; then
    echo -e "${RED}PostgreSQL is not running at $DB_HOST:$DB_PORT${NC}"
    exit 1
fi

# Create the test database if it doesn't exist
if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $TEST_DB_NAME; then
    echo -e "${YELLOW}Test database '$TEST_DB_NAME' already exists. Recreating...${NC}"
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "DROP DATABASE IF EXISTS $TEST_DB_NAME;"
fi

echo -e "${YELLOW}Creating test database '$TEST_DB_NAME'...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $TEST_DB_NAME;"

# Run the schema script to create tables
echo -e "${YELLOW}Creating schema in test database...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $TEST_DB_NAME -f db/schema.sql

# Run migrations
echo -e "${YELLOW}Running migrations...${NC}"
python db/run_migrations.py --host $DB_HOST --port $DB_PORT --dbname $TEST_DB_NAME --user $DB_USER --password $DB_PASSWORD

echo -e "${GREEN}Test database setup complete!${NC}"

# Optionally insert test data
echo -e "${YELLOW}Inserting test data...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $TEST_DB_NAME -c "
-- Insert test brand
INSERT INTO brands (id, name, description) 
VALUES ('test-brand-id', 'Test Brand', 'This is a test brand');

-- Insert test website
INSERT INTO websites (id, domain, url, is_flagged, analysis_status)
VALUES ('test-website-id', 'example.com', 'https://example.com', true, 'PENDING');

-- Insert test brand_custom_models
INSERT INTO brand_custom_models (id, brand_id, project_arn, status, status_message)
VALUES ('test-model-id', 'test-brand-id', 'test-project-arn', 'TRAINING_COMPLETED', 'Model training completed');
"

echo -e "${GREEN}Test data inserted!${NC}"
echo -e "${GREEN}Test database is ready for use.${NC}" 