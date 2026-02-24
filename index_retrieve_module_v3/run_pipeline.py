
import argparse
import logging
import sys
import importlib.util
from pathlib import Path

from pipeline.models import PipelineInput
from pipeline.pipeline import TestFixPipeline


def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from some libraries
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def load_config_from_file(config_file: str):
    """
    Load Config class from a Python file.
    
    Args:
        config_file: Path to config file (e.g., "config.py")
    
    Returns:
        Config instance
    """
    config_path = Path(config_file).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load module from file
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Get Config class and instantiate
    if not hasattr(config_module, 'Config'):
        raise ValueError(f"Config file must contain a 'Config' class: {config_path}")
    
    config = config_module.Config()
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run the test fix auxiliary information retrieval pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with default config.py
  python run_pipeline.py
  
  # Run with custom config file
  python run_pipeline.py --config-file my_config.py
  
  # Run with verbose logging
  python run_pipeline.py --verbose
  
  # Skip index rebuild for faster iteration (testing only)
  python run_pipeline.py --skip-rebuild
  
  # Skip git reset (testing only)
  python run_pipeline.py --skip-phase0
  
  # Specify custom output directory
  python run_pipeline.py --output-dir ./my_results

Environment Variables:
  OPENAI_API_KEY      Your OpenAI API key (required)
  OPENAI_BASE_URL     OpenAI API base URL (optional)
        """
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default='config.py',
        help='Path to config file (default: config.py)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='pipeline_results',
        help='Directory to save results (default: pipeline_results/)'
    )
    
    parser.add_argument(
        '--skip-rebuild',
        action='store_true',
        help='Skip index rebuild (for faster iteration during testing)'
    )
    
    parser.add_argument(
        '--skip-phase0',
        action='store_true',
        help='Skip git reset in Phase 0 (for testing)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--max-sql-rounds',
        type=int,
        default=3,
        help='Maximum SQL query rounds in Phase 2 (default: 3)'
    )
    
    parser.add_argument(
        '--max-rag-rounds',
        type=int,
        default=3,
        help='Maximum RAG search rounds in Phase 3 (default: 3)'
    )
    
    parser.add_argument(
        '--rag-top-k',
        type=int,
        default=8,
        help='Number of RAG results to retrieve (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config_file}")
        config = load_config_from_file(args.config_file)
        
        # Create pipeline input
        pipeline_input = PipelineInput.from_config(
            config,
            skip_rebuild=args.skip_rebuild,
            skip_phase0=args.skip_phase0,
            max_sql_rounds=args.max_sql_rounds,
            max_rag_rounds=args.max_rag_rounds,
            rag_top_k=args.rag_top_k
        )
        
        logger.info("Pipeline configuration:")
        logger.info(f"  Repository: {pipeline_input.repo_path}")
        logger.info(f"  Commit: {pipeline_input.error_commit_id}")
        logger.info(f"  Model: {pipeline_input.agent_model}")
        logger.info(f"  Skip rebuild: {pipeline_input.skip_rebuild}")
        logger.info(f"  Skip Phase 0: {pipeline_input.skip_phase0}")
        logger.info("")
        
        # Create and run pipeline
        pipeline = TestFixPipeline(pipeline_input)
        output = pipeline.run_pipeline()
        
        # Save results
        output_dir = Path(args.output_dir)
        result_file = pipeline.save_results(output, output_dir)
        
        # Display summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Final Status: {output.final_status.value}")
        logger.info(f"Final Reason: {output.final_reason}")
        logger.info(f"Total Duration: {output.total_duration_seconds:.2f} seconds")
        logger.info(f"Results saved to: {result_file}")
        
        if output.final_context:
            logger.info("")
            logger.info("Retrieved Context (preview):")
            logger.info("-" * 70)
            
            # Ensure context is a string
            context_str = str(output.final_context) if output.final_context else ""
            context_preview = context_str[:500]
            logger.info(context_preview)
            
            if len(context_str) > 500:
                logger.info("... (truncated, see output file for full context)")
        
        logger.info("=" * 70)
        
        # Return appropriate exit code
        from pipeline.models import PipelineStatus
        if output.final_status in [
            PipelineStatus.SUCCESS_PHASE_3,
            PipelineStatus.STOPPED_AT_PHASE_2
        ]:
            return 0
        else:
            return 1
    
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("")
        logger.error("Please ensure:")
        logger.error("  1. OPENAI_API_KEY environment variable is set")
        logger.error("  2. OPENAI_BASE_URL environment variable is set (if using custom endpoint)")
        logger.error("  3. Config file exists and contains a 'Config' class")
        return 1
    
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())

