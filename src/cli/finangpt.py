#!/usr/bin/env python3
"""FinanGPT Unified CLI - Phase 7: Unified Workflow & Automation

Single entry point for all FinanGPT operations:
- finangpt ingest: Ingest financial data
- finangpt transform: Transform data to DuckDB
- finangpt query: Execute one-shot queries
- finangpt chat: Interactive conversational mode
- finangpt status: Check data freshness and system status
- finangpt refresh: Full data refresh workflow
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import duckdb
from pymongo import MongoClient

from src.core.config_loader import load_config
from src.core.time_utils import parse_utc_timestamp


def get_status(config_path: Optional[str] = None) -> dict:
    """Get system status including data freshness and ticker count.

    Args:
        config_path: Optional path to config file

    Returns:
        Dictionary with status information
    """
    config = load_config(config_path)
    status = {
        "timestamp": datetime.now(UTC).isoformat(),
        "database": {},
        "data_freshness": {},
        "configuration": {},
    }

    # Check MongoDB connection
    try:
        client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=5000)
        db = client.get_default_database()
        if not db:
            db_name = config.mongo_uri.rsplit("/", 1)[-1]
            db = client[db_name]

        status["database"]["mongodb"] = "connected"

        # Get ticker count
        metadata_collection = db["ingestion_metadata"]
        unique_tickers = metadata_collection.distinct("ticker")
        status["database"]["ticker_count"] = len(unique_tickers)

        # Get data freshness summary
        freshness_data = {}
        for ticker in unique_tickers:
            most_recent = metadata_collection.find_one(
                {"ticker": ticker},
                sort=[("last_fetched", -1)]
            )
            if most_recent:
                last_fetched_str = most_recent.get("last_fetched")
                if last_fetched_str:
                    last_fetched = parse_utc_timestamp(last_fetched_str)
                    age_days = (datetime.now(UTC) - last_fetched).days
                    freshness_data[ticker] = {
                        "days_old": age_days,
                        "last_fetched": last_fetched_str,
                    }

        # Calculate freshness statistics
        if freshness_data:
            ages = [data["days_old"] for data in freshness_data.values()]
            status["data_freshness"]["average_age_days"] = sum(ages) / len(ages)
            status["data_freshness"]["oldest_age_days"] = max(ages)
            status["data_freshness"]["newest_age_days"] = min(ages)
            stale_count = sum(1 for age in ages if age >= config.auto_refresh_threshold_days)
            status["data_freshness"]["stale_ticker_count"] = stale_count
            status["data_freshness"]["stale_threshold_days"] = config.auto_refresh_threshold_days

        client.close()

    except Exception as e:
        status["database"]["mongodb"] = f"error: {e}"

    # Check DuckDB
    try:
        conn = duckdb.connect(config.duckdb_path, read_only=True)
        status["database"]["duckdb"] = "connected"

        # Get table counts
        tables = [
            "financials.annual",
            "financials.quarterly",
            "prices.daily",
            "dividends.history",
            "splits.history",
            "company.metadata",
            "company.peers",
            "ratios.financial",
            "user.portfolios",
        ]

        table_counts = {}
        for table in tables:
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                table_counts[table] = result[0] if result else 0
            except duckdb.CatalogException:
                table_counts[table] = "not found"

        status["database"]["table_counts"] = table_counts
        conn.close()

    except Exception as e:
        status["database"]["duckdb"] = f"error: {e}"

    # Configuration summary
    status["configuration"]["mongo_uri"] = config.mongo_uri
    status["configuration"]["ollama_url"] = config.ollama_url
    status["configuration"]["model"] = config.model_name
    status["configuration"]["duckdb_path"] = config.duckdb_path

    return status


def print_status(status: dict) -> None:
    """Pretty-print status information.

    Args:
        status: Status dictionary from get_status()
    """
    print("\n" + "=" * 70)
    print("FinanGPT System Status")
    print("=" * 70)
    print(f"\nTimestamp: {status['timestamp']}\n")

    # Database status
    print("📊 Database Status:")
    print(f"  MongoDB: {status['database'].get('mongodb', 'unknown')}")
    print(f"  DuckDB: {status['database'].get('duckdb', 'unknown')}")
    print(f"  Ticker Count: {status['database'].get('ticker_count', 0)}")

    # Data freshness
    if "data_freshness" in status and status["data_freshness"]:
        print("\n📅 Data Freshness:")
        freshness = status["data_freshness"]
        print(f"  Average Age: {freshness.get('average_age_days', 0):.1f} days")
        print(f"  Oldest: {freshness.get('oldest_age_days', 0)} days")
        print(f"  Newest: {freshness.get('newest_age_days', 0)} days")
        stale_count = freshness.get('stale_ticker_count', 0)
        total_tickers = status['database'].get('ticker_count', 0)
        print(f"  Stale Tickers: {stale_count}/{total_tickers} (>{freshness.get('stale_threshold_days', 7)} days)")

    # Table counts
    if "table_counts" in status["database"]:
        print("\n📁 Table Row Counts:")
        for table, count in status["database"]["table_counts"].items():
            print(f"  {table}: {count}")

    # Configuration
    print("\n⚙️  Configuration:")
    config = status["configuration"]
    print(f"  MongoDB: {config['mongo_uri']}")
    print(f"  DuckDB: {config['duckdb_path']}")
    print(f"  Ollama: {config['ollama_url']}")
    print(f"  Model: {config['model']}")

    print("\n" + "=" * 70 + "\n")


def run_ingest(args: argparse.Namespace) -> int:
    """Run data ingestion.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    cmd = ["python", "-m", "src.ingestion.ingest"]

    if args.tickers:
        cmd.extend(["--tickers", args.tickers])
    if args.tickers_file:
        cmd.extend(["--tickers-file", args.tickers_file])
    if args.refresh:
        cmd.append("--refresh")
    if args.refresh_days:
        cmd.extend(["--refresh-days", str(args.refresh_days)])
    if args.force:
        cmd.append("--force")

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_transform(args: argparse.Namespace) -> int:
    """Run data transformation.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    cmd = ["python", "-m", "src.transformation.transform"]

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_query(args: argparse.Namespace) -> int:
    """Run one-shot query.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    cmd = ["python", "-m", "src.query_engine.query"]

    if args.question:
        cmd.append(args.question)
    if args.skip_freshness_check:
        cmd.append("--skip-freshness-check")
    if args.no_chart:
        cmd.append("--no-chart")
    if args.no_formatting:
        cmd.append("--no-formatting")
    if args.template:
        cmd.extend(["--template", args.template])
    if args.template_params:
        cmd.extend(["--template-params", args.template_params])
    if args.list_templates:
        cmd.append("--list-templates")
    if args.debug:
        cmd.append("--debug")

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_chat(args: argparse.Namespace) -> int:
    """Run interactive chat.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    cmd = ["python", "-m", "src.ui.chat"]

    if args.skip_freshness_check:
        cmd.append("--skip-freshness-check")
    if args.debug:
        cmd.append("--debug")

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_status(args: argparse.Namespace) -> int:
    """Show system status.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    try:
        status = get_status(args.config)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print_status(status)
        return 0
    except Exception as e:
        print(f"Error getting status: {e}")
        return 1


def run_refresh(args: argparse.Namespace) -> int:
    """Run full refresh workflow.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    print("\n" + "=" * 70)
    print("Full Refresh Workflow")
    print("=" * 70 + "\n")

    # Step 1: Ingest
    print("Step 1: Ingesting data...")
    ingest_cmd = ["python", "-m", "src.ingestion.ingest", "--refresh"]

    if args.tickers:
        ingest_cmd.extend(["--tickers", args.tickers])
    if args.tickers_file:
        ingest_cmd.extend(["--tickers-file", args.tickers_file])
    if args.all_data_types:
        # This is the default behavior - fetch all data types
        pass

    result = subprocess.call(ingest_cmd)
    if result != 0:
        print("\n❌ Ingestion failed")
        return result

    # Step 2: Transform
    print("\nStep 2: Transforming data...")
    result = subprocess.call(["python", "-m", "src.transformation.transform"])
    if result != 0:
        print("\n❌ Transformation failed")
        return result

    # Step 3: Show status
    print("\nStep 3: Checking status...")
    status = get_status(args.config)
    print_status(status)

    print("\n✅ Full refresh completed successfully!\n")
    return 0


def main() -> int:
    """Main entry point for unified CLI.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="FinanGPT Unified CLI - Single entry point for all operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml file (default: config.yaml in current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest financial data")
    ingest_parser.add_argument("--tickers", type=str, help="Comma-separated ticker symbols")
    ingest_parser.add_argument("--tickers-file", type=str, help="Path to file with tickers")
    ingest_parser.add_argument("--refresh", action="store_true", help="Refresh mode (smart caching)")
    ingest_parser.add_argument("--refresh-days", type=int, help="Refresh threshold in days")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-fetch all data")

    # Transform command
    transform_parser = subparsers.add_parser("transform", help="Transform data to DuckDB")

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute one-shot query")
    query_parser.add_argument("question", nargs="?", help="Natural language question")
    query_parser.add_argument("--skip-freshness-check", action="store_true", help="Skip freshness check")
    query_parser.add_argument("--no-chart", action="store_true", help="Disable chart generation")
    query_parser.add_argument("--no-formatting", action="store_true", help="Disable financial formatting")
    query_parser.add_argument("--template", type=str, help="Use query template")
    query_parser.add_argument("--template-params", type=str, help="Template parameters")
    query_parser.add_argument("--list-templates", action="store_true", help="List available templates")
    query_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive conversational mode")
    chat_parser.add_argument("--skip-freshness-check", action="store_true", help="Skip freshness check")
    chat_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Refresh command
    refresh_parser = subparsers.add_parser("refresh", help="Full refresh workflow")
    refresh_parser.add_argument("--tickers", type=str, help="Comma-separated ticker symbols")
    refresh_parser.add_argument("--tickers-file", type=str, help="Path to file with tickers")
    refresh_parser.add_argument("--all-data-types", action="store_true", help="Refresh all data types (default)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate handler
    if args.command == "ingest":
        return run_ingest(args)
    elif args.command == "transform":
        return run_transform(args)
    elif args.command == "query":
        return run_query(args)
    elif args.command == "chat":
        return run_chat(args)
    elif args.command == "status":
        return run_status(args)
    elif args.command == "refresh":
        return run_refresh(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
