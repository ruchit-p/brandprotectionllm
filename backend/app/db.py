import os
import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from contextlib import contextmanager
from app.config import get_settings

# Load environment variables
load_dotenv()

settings = get_settings()

def get_db_connection():
    """
    Create a new database connection
    """
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        cursor_factory=RealDictCursor
    )
    conn.autocommit = True
    return conn

@contextmanager
def get_db_cursor():
    """
    Context manager for database cursor
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_qdrant_client():
    """
    Create and return a Qdrant client
    """
    try:
        client = QdrantClient(url=os.getenv("QDRANT_URL"))
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        raise

def setup_qdrant_collections():
    """
    Initialize Qdrant collections for vector storage
    """
    client = get_qdrant_client()
    
    collections = {
        "brand_logos": {
            "size": 1536,  # Claude embedding dimensions
            "distance": "Cosine",
            "description": "Brand logo embeddings"
        },
        "brand_images": {
            "size": 1536,
            "distance": "Cosine",
            "description": "Brand product image embeddings"
        },
        "site_html": {
            "size": 1536,
            "distance": "Cosine",
            "description": "HTML structure embeddings"
        },
        "site_text": {
            "size": 1536,
            "distance": "Cosine",
            "description": "Site text content embeddings"
        },
        "site_images": {
            "size": 1536,
            "distance": "Cosine",
            "description": "Site image embeddings"
        }
    }
    
    for name, config in collections.items():
        # Check if collection exists
        try:
            collection_info = client.get_collection(name)
            print(f"Collection {name} already exists")
        except Exception:
            # Create collection if it doesn't exist
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "size": config["size"],
                    "distance": config["distance"],
                },
                metadata={
                    "description": config["description"]
                }
            )
            print(f"Collection {name} created")

# Initialize Qdrant collections if this file is run directly
if __name__ == "__main__":
    setup_qdrant_collections()
