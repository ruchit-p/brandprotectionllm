# Brand Protection System

An AI-powered system to help companies identify and monitor websites that may be impersonating their brand or using their assets without permission. The system combines AI-powered analysis with advanced web crawling and similarity detection to provide comprehensive protection.

## Features

- **AI-Guided Onboarding**: LangChain with Claude 3.7 guides the conversation to collect brand information
- **Web Crawling**: Automated crawling with Firecrawl to establish baseline content and discover potential infringements
- **Multi-faceted Analysis**:
  - Text analysis using Claude
  - HTML similarity detection using JPlag
  - Image analysis with AWS Rekognition
  - Vector similarity search with Qdrant
- **Intuitive Dashboard**: Review flagged sites with detailed evidence

## Technology Stack

- **Frontend**: Next.js with Tailwind CSS
- **Backend**: Python with FastAPI
- **Database**: PostgreSQL (relational data) and Qdrant (vector database)
- **AI & Analysis**: Claude 3.7, AWS Rekognition, JPlag
- **Background Processing**: Celery with Redis for task queue and scheduling
- **Monitoring**: Flower for Celery task monitoring

## System Architecture

The system consists of the following components:

1. **Client Onboarding System**: Guided information collection using Claude 3.7
2. **Data Collection Engine**: Automated web crawling with Firecrawl
3. **Analysis Pipeline**: Multi-faceted detection using vector similarity, JPlag, and AWS Rekognition
4. **Admin Dashboard**: Centralized interface for reviewing and managing flagged sites
5. **Background Processing System**: Distributed task processing with Celery and Redis

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL
- Redis (for Celery task queue)
- Qdrant (local or cloud instance)
- Anthropic API key
- AWS account with Rekognition access
- Local Firecrawl instance

### Backend Setup

1. Navigate to the backend directory:

   ```
   cd backend
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the `.env.example` template and fill in your API keys and configuration.

5. Initialize the database:

   ```
   psql -U postgres -c "CREATE DATABASE brand_protection;"
   psql -U postgres -d brand_protection -f db/schema.sql
   ```

6. Run Redis for Celery task queue:

   ```
   docker run -d --name redis-brand-protection -p 6379:6379 redis:alpine
   ```

7. Run the server, Celery workers, and scheduler:

   ```
   # Terminal 1 - FastAPI server
   uvicorn app.main:app --reload

   # Terminal 2 - Celery worker for rekognition tasks
   cd backend
   ./run_worker.sh rekognition

   # Terminal 3 - Celery worker for analysis tasks
   cd backend
   ./run_worker.sh analysis

   # Terminal 4 - Celery worker for maintenance tasks
   cd backend
   ./run_worker.sh maintenance

   # Terminal 5 - Celery beat scheduler
   cd backend
   ./run_worker.sh beat

   # Terminal 6 - Flower monitoring UI (optional)
   cd backend
   ./run_worker.sh flower
   ```

Alternatively, you can use Docker Compose to run all services:

```
docker-compose up -d
```

### Frontend Setup

1. Navigate to the frontend directory:

   ```
   cd frontend
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Create a `.env.local` file with:

   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. Run the development server:
   ```
   npm run dev
   ```

## Usage

1. **Onboarding**: Start by entering your brand information through the AI-guided conversation.
2. **Upload Logo**: Upload your brand logo to enable image similarity detection.
3. **Dashboard**: Review flagged websites and examine evidence of potential infringement.
4. **Actions**: Confirm genuine infringements or dismiss false positives.

## License

[MIT](LICENSE)

## Acknowledgments

- Anthropic for Claude API
- AWS for Rekognition
- JPlag for code similarity detection

## Running Tests

To run tests for the backend:

```
cd backend
./run_tests.sh
```

To run specific test files:

```
./run_tests.sh tests/tasks/test_rekognition_tasks.py
```

## Background Processing

The system uses Celery with Redis for robust background task processing:

1. **Task Queues**:

   - Rekognition: AWS Rekognition tasks and custom model management
   - Analysis: Website analysis and similarity detection
   - Maintenance: System maintenance and monitoring

2. **Scheduled Tasks**:

   - Model status updates every 5 minutes
   - Result cleanup daily
   - Interrupted task recovery hourly
   - Database statistics updates daily
   - AWS permission verification daily

3. **Monitoring**:
   - Access Flower dashboard at http://localhost:5555 to monitor task status and performance
