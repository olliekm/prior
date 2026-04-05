import os
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Connection pool: min 1, max 10 connections
_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    """Lazy-initialize connection pool."""
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(1, 10, os.environ['DATABASE_URL'])
    return _pool


def get_conn():
    """Get a connection from the pool."""
    return _get_pool().getconn()


def release_conn(conn):
    """Return a connection to the pool."""
    _get_pool().putconn(conn)

def init_db():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id          TEXT PRIMARY KEY,
                    title       TEXT,
                    abstract    TEXT,
                    authors     TEXT[],
                    published   DATE,
                    categories  TEXT[],
                    embedding   vector(1536)
                );
            """)
        conn.commit()
    finally:
        release_conn(conn)


def embed(text: str) -> list[float]:
    """Embed a single text string."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call (much faster)."""
    if not texts:
        return []
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    # Response maintains input order
    return [item.embedding for item in response.data]


def upsert_papers(papers: list[dict]):
    """
    papers: list of dicts with keys matching the table columns.
    Uses batch embedding for efficiency (1 API call instead of N).
    """
    if not papers:
        return

    # Batch embed all papers in a single API call
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = embed_batch(texts)

    rows = [
        (
            p["id"],
            p["title"],
            p["abstract"],
            p.get("authors", []),
            p.get("published"),
            p.get("categories", []),
            vec
        )
        for p, vec in zip(papers, embeddings)
    ]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO papers (id, title, abstract, authors, published, categories, embedding)
                VALUES %s
                ON CONFLICT (id) DO NOTHING;
            """, rows)
        conn.commit()
    finally:
        release_conn(conn)

def vector_search(query: str, top_k: int = 15) -> list[dict]:
    query_vec = embed(query)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, abstract, authors, published, categories,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM papers
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vec, query_vec, top_k))

            cols = ["id", "title", "abstract", "authors", "published", "categories", "similarity"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_conn(conn)
