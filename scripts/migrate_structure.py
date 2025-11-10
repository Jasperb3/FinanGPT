#!/usr/bin/env python3
"""
Directory structure migration script for FinanGPT.

This script reorganizes the flat root structure into a logical src/ hierarchy
while maintaining backward compatibility through wrapper files.

Author: Enhancement Plan 4 - Phase 5
Created: 2025-11-10
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# File relocation mapping: (source, destination)
FILE_MOVES = [
    ("ingest.py", "src/ingestion/core.py"),
    ("transform.py", "src/transformation/core.py"),
    ("query.py", "src/query/executor.py"),
    ("chat.py", "src/query/chat.py"),
    ("resilience.py", "src/query/resilience.py"),
    ("visualize.py", "src/visualization/charts.py"),
    ("valuation.py", "src/intelligence/valuation.py"),
    ("analyst.py", "src/intelligence/analyst.py"),
    ("technical.py", "src/intelligence/technical.py"),
    ("query_history.py", "src/query/history.py"),
    ("query_planner.py", "src/query/planner.py"),
    ("error_handler.py", "src/intelligence/error_handler.py"),
    ("autocomplete.py", "src/intelligence/autocomplete.py"),
    ("config_loader.py", "src/utils/config.py"),
    ("peer_groups.py", "src/utils/peer_groups.py"),
    ("date_parser.py", "src/utils/date_parser.py"),
    ("time_utils.py", "src/utils/time_utils.py"),
]

# Import path replacements: (old_pattern, new_replacement)
IMPORT_REPLACEMENTS = [
    # Specific file imports (do these first for precision)
    (r'^from time_utils import', 'from src.utils.time_utils import'),
    (r'^import time_utils$', 'import src.utils.time_utils as time_utils'),
    (r'^from config_loader import', 'from src.utils.config import'),
    (r'^import config_loader$', 'import src.utils.config as config_loader'),
    (r'^from peer_groups import', 'from src.utils.peer_groups import'),
    (r'^from date_parser import', 'from src.utils.date_parser import'),
    (r'^from query_history import', 'from src.query.history import'),
    (r'^from query_planner import', 'from src.query.planner import'),
    (r'^from error_handler import', 'from src.intelligence.error_handler import'),
    (r'^from autocomplete import', 'from src.intelligence.autocomplete import'),
    (r'^from resilience import', 'from src.query.resilience import'),
    (r'^from visualize import', 'from src.visualization.charts import'),
    (r'^from valuation import', 'from src.intelligence.valuation import'),
    (r'^from analyst import', 'from src.intelligence.analyst import'),
    (r'^from technical import', 'from src.intelligence.technical import'),

    # Directory imports (existing src/ structure updates)
    (r'from src\.ingest\.', 'from src.ingestion.'),
    (r'import src\.ingest\.', 'import src.ingestion.'),
    (r'from src\.transform\.', 'from src.transformation.'),
    (r'import src\.transform\.', 'import src.transformation.'),
]

WRAPPER_TEMPLATE = '''#!/usr/bin/env python3
"""
Backward-compatible wrapper for {original_name}.

This file maintains compatibility with legacy scripts and imports.
New code should import from {new_module} directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from {original_name} is deprecated. "
    "Use 'from {new_module} import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from {new_module} import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from {new_module} import main
    sys.exit(main())
'''


def create_init_files():
    """Create __init__.py files in all new directories."""
    init_dirs = [
        "src",
        "src/ingestion",
        "src/transformation",
        "src/query",
        "src/intelligence",
        "src/visualization",
        "src/utils",
        "src/data",
    ]

    for dir_path in init_dirs:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f'"""FinanGPT {dir_path.split("/")[-1]} module."""\n')
            print(f"✓ Created {init_file}")


def update_imports_in_file(file_path: Path, dry_run: bool = False) -> int:
    """Update import statements in a single file."""
    if not file_path.exists() or file_path.suffix != '.py':
        return 0

    content = file_path.read_text()
    original_content = content
    lines = content.split('\n')
    changes = 0

    # Process each line
    new_lines = []
    for line in lines:
        new_line = line

        # Apply import replacements
        for old_pattern, new_replacement in IMPORT_REPLACEMENTS:
            if re.match(old_pattern, line.strip()):
                new_line = re.sub(old_pattern, new_replacement, line)
                if new_line != line:
                    changes += 1
                    break

        new_lines.append(new_line)

    new_content = '\n'.join(new_lines)

    if new_content != original_content:
        if not dry_run:
            file_path.write_text(new_content)
        print(f"  Updated {changes} import(s) in {file_path}")
        return changes

    return 0


def move_file(source: str, destination: str, dry_run: bool = False) -> bool:
    """Move a file to its new location."""
    source_path = Path(source)
    dest_path = Path(destination)

    if not source_path.exists():
        print(f"⚠ Source not found: {source}")
        return False

    # Create destination directory
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY RUN] Would move: {source} → {destination}")
        return True

    # Move file
    shutil.move(str(source_path), str(dest_path))
    print(f"✓ Moved: {source} → {destination}")
    return True


def create_wrapper(original_file: str, new_module: str, dry_run: bool = False) -> bool:
    """Create backward-compatible wrapper file."""
    original_name = Path(original_file).stem

    # Convert path to module notation
    # src/ingestion/core.py → src.ingestion.core
    module_path = new_module.replace('.py', '').replace('/', '.')

    wrapper_content = WRAPPER_TEMPLATE.format(
        original_name=original_file,
        new_module=module_path
    )

    wrapper_path = Path(original_file)

    if dry_run:
        print(f"[DRY RUN] Would create wrapper: {wrapper_path}")
        return True

    wrapper_path.write_text(wrapper_content)
    print(f"✓ Created wrapper: {wrapper_path}")
    return True


def update_all_imports(root_dir: Path, dry_run: bool = False) -> int:
    """Update imports in all Python files."""
    total_changes = 0

    print("\n=== Updating imports in all Python files ===")

    # Find all Python files
    for py_file in root_dir.rglob("*.py"):
        # Skip __pycache__ and .venv
        if '__pycache__' in str(py_file) or '.venv' in str(py_file):
            continue

        changes = update_imports_in_file(py_file, dry_run)
        total_changes += changes

    return total_changes


def verify_structure():
    """Verify the new structure is correct."""
    required_files = [
        "src/ingestion/core.py",
        "src/transformation/core.py",
        "src/query/executor.py",
        "src/query/chat.py",
        "src/intelligence/valuation.py",
        "src/intelligence/analyst.py",
        "src/intelligence/technical.py",
        "src/visualization/charts.py",
        "src/utils/config.py",
        "src/utils/peer_groups.py",
    ]

    print("\n=== Verifying structure ===")
    all_good = True

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_good = False

    return all_good


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate FinanGPT directory structure")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    parser.add_argument('--verify-only', action='store_true', help="Only verify structure, don't migrate")
    args = parser.parse_args()

    root_dir = Path.cwd()

    if args.verify_only:
        success = verify_structure()
        sys.exit(0 if success else 1)

    print("=" * 70)
    print("FinanGPT Directory Structure Migration")
    print("=" * 70)

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made\n")
    else:
        print("\n⚠️  LIVE MODE - Files will be moved and modified\n")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    # Step 1: Create __init__.py files
    print("\n=== Step 1: Creating __init__.py files ===")
    create_init_files()

    # Step 2: Move files to new locations
    print("\n=== Step 2: Moving files to new structure ===")
    for source, destination in FILE_MOVES:
        move_file(source, destination, dry_run=args.dry_run)

    # Step 3: Update imports in all files
    print("\n=== Step 3: Updating imports ===")
    total_changes = update_all_imports(root_dir, dry_run=args.dry_run)
    print(f"\nTotal import updates: {total_changes}")

    # Step 4: Create backward-compatible wrappers
    print("\n=== Step 4: Creating backward-compatible wrappers ===")
    for source, destination in FILE_MOVES:
        # Skip finangpt.py (stays in root)
        if source == "finangpt.py":
            continue

        create_wrapper(source, destination, dry_run=args.dry_run)

    # Step 5: Verify structure
    if not args.dry_run:
        if verify_structure():
            print("\n✅ Migration completed successfully!")
        else:
            print("\n❌ Migration completed with errors. Please review.")
            sys.exit(1)
    else:
        print("\n✅ Dry run completed. Review changes above.")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Test CLI commands: python finangpt.py status")
    print("2. Run test suite: pytest tests/ -v")
    print("3. Update documentation")
    print("=" * 70)


if __name__ == "__main__":
    main()
