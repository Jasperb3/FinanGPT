"""
Streaming transformation module for MongoDB → DuckDB data transfer.

This module implements chunked streaming to prevent memory overflow when
transforming large datasets from MongoDB to DuckDB.

Features:
- Memory-efficient chunked processing
- Configurable chunk sizes
- Progress tracking support
- Error handling with partial success

Author: FinanGPT Enhancement Plan 3
Created: 2025-11-09
"""

from typing import Iterator, Callable, Dict, Any, List, Optional
from pymongo.collection import Collection
import pandas as pd
import duckdb
import logging
from datetime import datetime

# Default chunk size - tested to balance memory vs. speed
DEFAULT_CHUNK_SIZE = 1000


def stream_documents(
    collection: Collection,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    query_filter: Optional[Dict[str, Any]] = None
) -> Iterator[List[Dict[str, Any]]]:
    """
    Stream MongoDB documents in chunks to prevent memory overflow.

    Args:
        collection: MongoDB collection to stream from
        chunk_size: Number of documents per chunk
        query_filter: Optional MongoDB query filter

    Yields:
        Lists of document dictionaries (chunk size or smaller for last chunk)

    Example:
        >>> for chunk in stream_documents(db['raw_annual'], chunk_size=500):
        ...     print(f"Processing {len(chunk)} documents...")
        ...     df = prepare_dataframe(chunk)
    """
    cursor = collection.find(query_filter or {}, batch_size=chunk_size)
    chunk = []

    for doc in cursor:
        chunk.append(doc)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    # Yield remaining documents
    if chunk:
        yield chunk


def transform_with_streaming(
    collection: Collection,
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    schema: str,
    prepare_func: Callable[[List[Dict]], pd.DataFrame],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Transform MongoDB collection to DuckDB table using streaming.

    This function prevents memory overflow by processing documents in chunks
    rather than loading the entire collection into memory.

    Args:
        collection: MongoDB collection to read from
        conn: DuckDB connection to write to
        table_name: Target table name (e.g., "annual", "quarterly")
        schema: DuckDB schema name (e.g., "financials", "prices")
        prepare_func: Function to convert documents to DataFrame
        chunk_size: Number of documents per chunk
        logger: Optional logger for tracking progress

    Returns:
        Total number of rows inserted

    Example:
        >>> from transform import prepare_annual_dataframe
        >>> rows = transform_with_streaming(
        ...     db['raw_annual'],
        ...     conn,
        ...     'annual',
        ...     'financials',
        ...     prepare_annual_dataframe,
        ...     chunk_size=1000
        ... )
        >>> print(f"Inserted {rows} rows")
    """
    total_rows = 0
    chunk_count = 0

    for chunk_docs in stream_documents(collection, chunk_size):
        chunk_count += 1

        # Prepare DataFrame from chunk
        chunk_df = prepare_func(chunk_docs)

        if chunk_df.empty:
            if logger:
                logger.warning(f"Empty DataFrame from chunk {chunk_count}")
            continue

        # Upsert chunk to DuckDB
        try:
            rows = upsert_dataframe(conn, chunk_df, table_name, schema)
            total_rows += rows

            if logger:
                logger.info(
                    f"Chunk {chunk_count}: {rows} rows inserted "
                    f"(total: {total_rows})"
                )

        except Exception as e:
            if logger:
                logger.error(f"Error inserting chunk {chunk_count}: {e}")
            # Continue processing remaining chunks
            continue

    return total_rows


def upsert_dataframe(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
    schema: str
) -> int:
    """
    Upsert DataFrame into DuckDB table.

    Uses DELETE + INSERT pattern for idempotency. If table doesn't exist,
    creates it automatically from DataFrame schema.

    Args:
        conn: DuckDB connection
        df: DataFrame to insert
        table_name: Table name (without schema)
        schema: Schema name

    Returns:
        Number of rows inserted

    Example:
        >>> df = pd.DataFrame({'ticker': ['AAPL'], 'revenue': [394000000000]})
        >>> rows = upsert_dataframe(conn, df, 'annual', 'financials')
    """
    if df.empty:
        return 0

    full_table_name = f"{schema}.{table_name}"

    # Create schema if not exists
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    # Create table if not exists (using LIMIT 0 to create empty table with schema)
    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} AS
            SELECT * FROM df LIMIT 0
        """)
    except Exception as e:
        # Table might already exist
        pass

    # Delete existing rows for these ticker+date combinations
    if 'ticker' in df.columns and 'date' in df.columns:
        # Get unique ticker+date pairs
        pairs = df[['ticker', 'date']].drop_duplicates()

        for _, row in pairs.iterrows():
            ticker = row['ticker']
            date = row['date']

            conn.execute(f"""
                DELETE FROM {full_table_name}
                WHERE ticker = ? AND date = ?
            """, [ticker, date])

    # Insert new data
    conn.execute(f"""
        INSERT INTO {full_table_name}
        SELECT * FROM df
    """)

    return len(df)


def get_collection_stats(collection: Collection) -> Dict[str, Any]:
    """
    Get statistics about MongoDB collection for streaming optimization.

    Args:
        collection: MongoDB collection

    Returns:
        Dictionary with count, avg_size, estimated_memory

    Example:
        >>> stats = get_collection_stats(db['raw_annual'])
        >>> print(f"Estimated memory: {stats['estimated_memory_mb']:.2f} MB")
    """
    # Get document count
    count = collection.count_documents({})

    if count == 0:
        return {
            'count': 0,
            'avg_size_bytes': 0,
            'estimated_memory_mb': 0
        }

    # Sample a few documents to estimate size
    sample = list(collection.aggregate([{'$sample': {'size': min(100, count)}}]))

    # Estimate average document size (rough approximation)
    import sys
    total_size = sum(sys.getsizeof(str(doc)) for doc in sample)
    avg_size = total_size / len(sample) if sample else 0

    # Estimate total memory if loaded all at once
    estimated_memory = (count * avg_size) / (1024 * 1024)  # Convert to MB

    return {
        'count': count,
        'avg_size_bytes': avg_size,
        'estimated_memory_mb': estimated_memory
    }


def recommend_chunk_size(
    collection: Collection,
    max_memory_mb: int = 500
) -> int:
    """
    Recommend optimal chunk size based on collection statistics.

    Args:
        collection: MongoDB collection
        max_memory_mb: Maximum memory to use per chunk (MB)

    Returns:
        Recommended chunk size

    Example:
        >>> chunk_size = recommend_chunk_size(db['raw_annual'], max_memory_mb=500)
        >>> print(f"Using chunk size: {chunk_size}")
    """
    stats = get_collection_stats(collection)

    if stats['avg_size_bytes'] == 0:
        return DEFAULT_CHUNK_SIZE

    # Calculate chunk size to stay under memory limit
    max_memory_bytes = max_memory_mb * 1024 * 1024
    recommended = int(max_memory_bytes / stats['avg_size_bytes'])

    # Clamp to reasonable range
    return max(100, min(recommended, 5000))
