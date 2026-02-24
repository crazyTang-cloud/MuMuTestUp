#!/usr/bin/env python3
"""
Command-line entry point for building the Java code index.

Usage:
    python build_index.py --project-root /path/to/java/project
    
Environment variables required:
    OPENAI_API_KEY: Your OpenAI API key
    OPENAI_BASE_URL: OpenAI API base URL (optional, defaults to https://api.openai.com/v1)
"""
import argparse
import logging
import sys
from pathlib import Path

from indexer.config import IndexerConfig
from indexer.builder import IndexBuilder


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from some libraries
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build index for Java code repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for a Java project
  python build_index.py --project-root /path/to/java/project
  
  # Build with verbose logging
  python build_index.py --project-root /path/to/java/project --verbose
  
  # Append to existing index (don't rebuild)
  python build_index.py --project-root /path/to/java/project --no-rebuild

  # Index a specific Maven module
  python build_index.py --project-root /path/to/java/project --module order-service
  
  # Skip vector indexing (fast mode for testing)
  python build_index.py --project-root /path/to/java/project --skip-vector

Environment Variables:
  OPENAI_API_KEY      Your OpenAI API key (required)
  OPENAI_BASE_URL     OpenAI API base URL (optional)
  JAVA_PROJECT_ROOT   Default project root if --project-root not specified
        """
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        help='Root directory of the Java project to index'
    )

    parser.add_argument(
        '--module',
        type=str,
        default=None,
        help=(
            'Only index a specific Maven module name '
            '(e.g., "order-service"). '
            'Use "root" to mean the root module in a single-module project. '
            'If omitted, all modules will be indexed.'
        ),
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        default=True,
        help='Clear existing index before building (default: True)'
    )
    
    parser.add_argument(
        '--no-rebuild',
        action='store_false',
        dest='rebuild',
        help='Append to existing index instead of rebuilding'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--skip-vector',
        action='store_true',
        help='Skip vector indexing (ChromaDB). Only build SQLite index for fast testing.'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = IndexerConfig.from_env(project_root=args.project_root)
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  Project root: {config.project_root}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  OpenAI base URL: {config.openai_base_url}")
        logger.info("")
        
        # Create and run builder
        builder = IndexBuilder(config)
        builder.build(rebuild=args.rebuild, target_module=args.module, skip_vector=args.skip_vector)
        
        logger.info("")
        logger.info("✓ Index build completed successfully!")
        logger.info(f"  SQLite database: {config.sqlite_path}")
        if not args.skip_vector:
            logger.info(f"  ChromaDB: {config.chroma_path}")
        else:
            logger.info(f"  ChromaDB: Skipped (--skip-vector)")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("")
        logger.error("Please ensure:")
        logger.error("  1. OPENAI_API_KEY environment variable is set")
        logger.error("  2. Project root is specified via --project-root or JAVA_PROJECT_ROOT")
        return 1
        
    except Exception as e:
        logger.error(f"Error during index build: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())

