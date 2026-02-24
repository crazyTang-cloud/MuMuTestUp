"""
Index builder wrapper for the pipeline.

Provides simplified interface for building lightweight (SQL-only) 
and full (SQL + vector) indexes with automatic cleanup.
"""
import logging
import shutil
from pathlib import Path
from typing import Optional, List

from indexer.config import IndexerConfig
from indexer.builder import IndexBuilder

logger = logging.getLogger(__name__)


class PipelineIndexBuilder:
    """
    Wrapper around IndexBuilder with pipeline-specific conveniences.
    
    Handles index cleanup and rebuilding for consistent pipeline execution.
    """
    
    def __init__(self, repo_path: str, output_dir: Optional[Path] = None):
        """
        Initialize pipeline index builder.
        
        Args:
            repo_path: Path to Java repository
            output_dir: Optional custom output directory (default: ./index_data)
        """
        self.repo_path = Path(repo_path).resolve()
        self.output_dir = output_dir or Path.cwd() / "index_data"
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        logger.info(f"Pipeline index builder initialized for: {self.repo_path}")
    
    def clean_index_data(self):
        """
        Clean up existing index data directory.
        
        Removes SQLite database and ChromaDB directory to ensure fresh build.
        """
        if self.output_dir.exists():
            logger.info(f"Cleaning index data directory: {self.output_dir}")
            
            try:
                # Remove SQLite database
                sqlite_path = self.output_dir / "assets.db"
                if sqlite_path.exists():
                    sqlite_path.unlink()
                    logger.debug("Removed assets.db")
                
                # Remove ChromaDB directory
                chroma_path = self.output_dir / "chroma_db"
                if chroma_path.exists():
                    shutil.rmtree(chroma_path)
                    logger.debug("Removed chroma_db directory")
                
                logger.info("Index data cleaned successfully")
                
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
                # Continue anyway
    
    def build_lightweight_index(
        self,
        skip_rebuild: bool = False,
        target_module: Optional[str] = None,
        verbose: bool = False
    ) -> bool:
        """
        Build lightweight index (SQLite only, no vector embeddings).
        
        This is used in Phase 2 for fast metadata-only indexing.
        
        Args:
            skip_rebuild: If True, don't clean existing data
            target_module: Optional specific module to index
            verbose: Enable verbose logging
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info("Building LIGHTWEIGHT index (SQL only, no embeddings)")
        logger.info("=" * 70)
        
        try:
            # Clean up unless skipping rebuild
            if not skip_rebuild:
                self.clean_index_data()
            
            # Load indexer configuration from main config.py
            config = IndexerConfig.from_main_config(
                project_root=str(self.repo_path),
                output_dir=str(self.output_dir)
            )
            
            # Create index builder
            builder = IndexBuilder(config)
            
            # Build with skip_vector=True for speed
            builder.build(
                rebuild=not skip_rebuild,
                target_module=target_module,
                skip_vector=True
            )
            
            logger.info("Lightweight index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build lightweight index: {e}", exc_info=verbose)
            return False
    
    def build_full_index(
        self,
        skip_rebuild: bool = False,
        target_module: Optional[str] = None,
        verbose: bool = False,
        embed_methods_only: bool = True
    ) -> bool:
        """
        Build full index (SQLite + vector embeddings).
        
        This is used in Phase 3 for semantic RAG search.
        
        Args:
            skip_rebuild: If True, don't clean existing data
            target_module: Optional specific module to index
            verbose: Enable verbose logging
            embed_methods_only: If True, only embed methods (not fields) for cost savings
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info("Building FULL index (SQL + vector embeddings)")
        logger.info("=" * 70)
        
        try:
            # Clean up unless skipping rebuild
            if not skip_rebuild:
                self.clean_index_data()
            
            # Load indexer configuration from main config.py
            config = IndexerConfig.from_main_config(
                project_root=str(self.repo_path),
                output_dir=str(self.output_dir)
            )
            
            # Create index builder
            builder = IndexBuilder(config)
            
            # Build with full vector embeddings
            # Note: For cost savings, we could skip field embeddings
            # by modifying the builder logic, but for now we build both
            logger.info("Building full index with embeddings...")
            if embed_methods_only:
                logger.info("Cost optimization: Only embedding methods table")
            
            builder.build(
                rebuild=not skip_rebuild,
                target_module=target_module,
                skip_vector=False
            )
            
            logger.info("Full index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build full index: {e}", exc_info=verbose)
            return False
    
    def build_incremental_module_embedding(
        self,
        module_name: str,
        verbose: bool = False
    ) -> bool:
        """
        Build vector embeddings for a single module incrementally.
        
        This does NOT rebuild SQLite (assumes it already exists from Phase 2).
        This does NOT re-parse files (reads from existing SQLite).
        This does NOT clear existing vector embeddings (incremental add).
        
        Args:
            module_name: Name of the module to embed
            verbose: Enable verbose logging
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Building incremental embeddings for module: {module_name}")
        
        try:
            # Load indexer configuration from main config.py
            config = IndexerConfig.from_main_config(
                project_root=str(self.repo_path),
                output_dir=str(self.output_dir)
            )
            
            # Directly read methods from SQLite (no re-parsing)
            from indexer.storage import SQLiteStorage
            from indexer.vector_store import VectorStore
            from indexer.models import MethodInfo
            
            sqlite_path = self.output_dir / "assets.db"
            if not sqlite_path.exists():
                logger.error(f"SQLite database not found: {sqlite_path}")
                return False
            
            # Read methods for this module from SQLite
            with SQLiteStorage(sqlite_path) as storage:
                cursor = storage.conn.cursor()
                cursor.execute("""
                    SELECT id, module_name, file_path, class_name, method_name,
                           signature, javadoc, body, start_line, end_line, is_test, imports
                    FROM methods
                    WHERE module_name = ? AND is_test = 0
                """, (module_name,))
                
                rows = cursor.fetchall()
                
                if not rows:
                    logger.warning(f"No methods found for module: {module_name}")
                    return True  # Not an error, just nothing to embed
                
                # Convert rows to MethodInfo objects
                methods = []
                for row in rows:
                    row_dict = dict(row)
                    # Parse imports if it's JSON
                    imports = row_dict.get('imports')
                    if imports:
                        try:
                            import json
                            imports = json.loads(imports)
                        except:
                            imports = None
                    
                    method = MethodInfo(
                        id=row_dict['id'],
                        module_name=row_dict['module_name'],
                        file_path=row_dict['file_path'],
                        class_name=row_dict['class_name'],
                        method_name=row_dict['method_name'],
                        signature=row_dict['signature'],
                        javadoc=row_dict.get('javadoc'),
                        body=row_dict['body'],
                        start_line=row_dict['start_line'],
                        end_line=row_dict['end_line'],
                        is_test=bool(row_dict['is_test']),
                        imports=imports
                    )
                    methods.append(method)
                
                logger.info(f"Loaded {len(methods)} methods from SQLite for module: {module_name}")
            
            # Build vector embeddings ONLY (no SQLite operations)
            vector_store = VectorStore(
                chroma_path=self.output_dir / "chroma_db",
                collection_name="java_methods",
                openai_api_key=config.openai_api_key,
                openai_base_url=config.openai_base_url
            )
            
            # Check for existing IDs to avoid duplicates
            try:
                collection = vector_store.vector_store._collection
                result = collection.get(include=[])
                existing_ids = set(result['ids']) if result and 'ids' in result else set()
                
                if existing_ids:
                    original_count = len(methods)
                    methods = [m for m in methods if m.id not in existing_ids]
                    logger.info(f"Filtered {original_count - len(methods)} already-embedded methods")
            except Exception as e:
                logger.debug(f"Could not check existing IDs: {e}")
            
            if not methods:
                logger.info(f"All methods from {module_name} already embedded")
                return True
            
            # Embed methods with larger batch size for speed
            from tqdm import tqdm
            logger.info(f"Embedding {len(methods)} methods from module: {module_name}")
            
            # Increased batch size to 500 for faster embedding
            # OpenAI API can handle larger batches efficiently
            batch_size = 150
            logger.info(f"Using batch size: {batch_size} (optimized for speed)")
            
            with tqdm(total=len(methods), desc=f"Embedding {module_name}", unit="method") as pbar:
                for i in range(0, len(methods), batch_size):
                    batch = methods[i:i + batch_size]
                    vector_store.add_methods(batch)
                    pbar.update(len(batch))
            
            logger.info(f"✓ Incremental embeddings built for module: {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build incremental embeddings for {module_name}: {e}", exc_info=verbose)
            return False
    
    def get_available_modules(self) -> List[str]:
        """
        Get list of available modules in the repository.
        
        Returns:
            List of module names
        """
        try:
            config = IndexerConfig.from_env(project_root=str(self.repo_path))
            from indexer.file_walker import JavaFileWalker
            
            walker = JavaFileWalker(
                project_root=config.project_root,
                ignored_dirs=config.ignored_dirs
            )
            
            # Get modules (returns dict {module_name: module_path})
            modules = walker._find_modules()
            module_names = list(modules.keys())
            
            logger.info(f"Available modules: {module_names}")
            return module_names
            
        except Exception as e:
            logger.error(f"Failed to get available modules: {e}")
            return []
    
    def check_index_exists(self, require_vectors: bool = False) -> bool:
        """
        Check if index exists and is valid.
        
        Args:
            require_vectors: If True, also check for ChromaDB
        
        Returns:
            True if index exists, False otherwise
        """
        sqlite_path = self.output_dir / "assets.db"
        
        if not sqlite_path.exists():
            return False
        
        if require_vectors:
            chroma_path = self.output_dir / "chroma_db"
            if not chroma_path.exists():
                return False
        
        return True

