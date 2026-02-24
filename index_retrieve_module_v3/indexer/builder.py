"""
Main index builder that coordinates all components.
"""
import logging
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

from .config import IndexerConfig
from .file_walker import JavaFileWalker
from .java_parser import JavaParser
from .storage import SQLiteStorage
from .vector_store import VectorStore
from .models import MethodInfo, FieldInfo

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Coordinates the entire indexing process:
    1. Walk through Java project files
    2. Parse each file to extract methods
    3. Store in SQLite (cold storage)
    4. Create vector embeddings in ChromaDB (hot storage)
    """
    
    def __init__(self, config: IndexerConfig):
        self.config = config
        
        # Initialize components
        self.file_walker = JavaFileWalker(
            project_root=config.project_root,
            ignored_dirs=config.ignored_dirs
        )
        self.java_parser = JavaParser()
        
    def build(self, rebuild: bool = True, target_module: str | None = None, skip_vector: bool = False):
        """
        Build the complete index.
        
        Args:
            rebuild: If True, clear existing data before building
            target_module: If specified, only index this module
            skip_vector: If True, skip vector indexing (only build SQLite for fast testing)
        """
        logger.info("=" * 70)
        logger.info("Starting index build process")
        logger.info("=" * 70)
        logger.info(f"Project root: {self.config.project_root}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("")
        
        start_time = time.time()
        
        # Step 1: Walk through project files
        logger.info("Step 1: Discovering Java files...")
        java_files = self.file_walker.walk()

        # Optional: filter by target module (e.g., 'root' or a specific sub-module name)
        if target_module is not None:
            original_count = len(java_files)
            java_files = [
                (path, module_name)
                for (path, module_name) in java_files
                if module_name == target_module
            ]
            logger.info(
                f"Filtering by module '{target_module}': "
                f"{len(java_files)} / {original_count} files will be indexed"
            )

        if not java_files:
            if target_module is not None:
                logger.warning(
                    f"No Java files found for module '{target_module}'. "
                    f"Check module name or project structure."
                )
            else:
                logger.warning("No Java files found in the project!")
            return
        
        logger.info("")
        
        # Step 2: Parse all files
        logger.info("Step 2: Parsing Java files and extracting methods and fields...")
        all_methods, all_fields = self._parse_all_files(java_files)
        
        if not all_methods and not all_fields:
            logger.warning("No methods or fields extracted from files!")
            return
        
        logger.info(f"Total methods extracted: {len(all_methods)}")
        logger.info(f"Total fields extracted: {len(all_fields)}")
        logger.info("")
        
        # Step 3: Store in SQLite
        logger.info("Step 3: Storing methods and fields in SQLite database...")
        self._store_in_sqlite(all_methods, rebuild)
        self._store_fields_in_sqlite(all_fields, rebuild)
        logger.info("")
        
        # Step 4: Create vector embeddings (optional)
        if skip_vector:
            logger.info("Step 4: Vector indexing - SKIPPED")
            logger.info("  (--skip-vector enabled for faster iteration)")
            logger.info("")
        else:
            logger.info("Step 4: Building vector index with OpenAI embeddings...")
            self._build_vector_index(all_methods, rebuild)
            self._build_field_vector_index(all_fields, rebuild)
            logger.info("")
        
        # Step 5: Display statistics
        elapsed_time = time.time() - start_time
        self._display_statistics(elapsed_time, skip_vector=skip_vector)
        
        logger.info("=" * 70)
        logger.info("Index build completed successfully!")
        logger.info("=" * 70)
    
    def _parse_all_files(self, java_files: List[tuple]) -> tuple[List[MethodInfo], List[FieldInfo]]:
        """
        Parse all Java files and extract methods and fields.
        
        Args:
            java_files: List of (file_path, module_name) tuples
            
        Returns:
            Tuple of (all_methods, all_fields) - Lists of MethodInfo and FieldInfo objects
        """
        all_methods = []
        all_fields = []
        failed_files = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(java_files), desc="Parsing files", unit="file") as pbar:
            for file_path, module_name in java_files:
                try:
                    # Get relative path
                    relative_path = self.file_walker.get_relative_path(file_path)
                    
                    # Check if test file
                    is_test = self.file_walker.is_test_file(file_path)
                    
                    # Parse the file (now returns methods and fields)
                    methods, fields = self.java_parser.parse_file(
                        file_path=file_path,
                        module_name=module_name,
                        relative_path=relative_path,
                        is_test=is_test
                    )
                    
                    all_methods.extend(methods)
                    all_fields.extend(fields)
                    
                    # Log details for each method
                    for method in methods:
                        test_marker = "[TEST]" if method.is_test else "[PROD]"
                        logger.debug(
                            f"  ✓ {test_marker} Method: {method.class_name}.{method.method_name} "
                            f"[lines {method.start_line}-{method.end_line}]"
                        )
                    
                    # Log details for each field
                    for field in fields:
                        test_marker = "[TEST]" if field.is_test else "[PROD]"
                        logger.debug(
                            f"  ✓ {test_marker} Field: {field.class_name}.{field.field_name} "
                            f"[lines {field.start_line}-{field.end_line}]"
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")
                    failed_files.append((file_path, str(e)))
                
                pbar.update(1)
        
        # Log summary
        if failed_files:
            logger.warning(f"Failed to parse {len(failed_files)} files:")
            for file_path, error in failed_files[:10]:  # Show first 10
                logger.warning(f"  - {file_path}: {error}")
            if len(failed_files) > 10:
                logger.warning(f"  ... and {len(failed_files) - 10} more")
        
        return all_methods, all_fields
    
    def _store_in_sqlite(self, methods: List[MethodInfo], rebuild: bool):
        """
        Store methods in SQLite database.
        
        Args:
            methods: List of MethodInfo objects
            rebuild: Whether to clear existing data
        """
        with SQLiteStorage(self.config.sqlite_path) as storage:
            if rebuild:
                logger.info("Clearing existing SQLite data...")
                storage.clear_all()
            
            logger.info(f"Inserting {len(methods)} methods into SQLite...")
            storage.insert_methods_batch(methods, batch_size=100)
            
            # Get statistics
            stats = storage.get_statistics()
            logger.info(f"  ✓ Production methods: {stats['production_methods']}")
            logger.info(f"  ✓ Test methods: {stats['test_methods']}")
            logger.info(f"  ✓ Total classes: {stats['classes']}")
    
    def _build_vector_index(self, methods: List[MethodInfo], rebuild: bool):
        """
        Build vector index using ChromaDB and OpenAI embeddings.
        
        Args:
            methods: List of MethodInfo objects
            rebuild: Whether to clear existing data
        """
        vector_store = VectorStore(
            chroma_path=self.config.chroma_path,
            collection_name=self.config.chroma_collection_name,
            openai_api_key=self.config.openai_api_key,
            openai_base_url=self.config.openai_base_url
        )
        
        if rebuild:
            logger.info("Clearing existing vector store data...")
            vector_store.clear_all()
        else:
            logger.info("Incremental mode: keeping existing embeddings, adding new ones")
            # Filter out methods that might already be embedded
            # This prevents duplicate ID errors
            existing_ids = self._get_existing_vector_ids(vector_store)
            if existing_ids:
                original_count = len(methods)
                methods = [m for m in methods if m.id not in existing_ids]
                logger.info(f"Filtered {original_count - len(methods)} already-embedded methods")
        
        if not methods:
            logger.info("No new methods to embed (all already exist)")
            stats = vector_store.get_statistics()
            logger.info(f"  ✓ Total vectors: {stats['total_vectors']}")
            return
        
        # Count test vs production methods
        test_count = sum(1 for m in methods if m.is_test)
        prod_count = len(methods) - test_count
        
        # Add methods with progress bar
        logger.info(f"Creating embeddings for {len(methods)} methods...")
        logger.info(f"  - Production: {prod_count}, Test: {test_count}")
        logger.info("  Note: Test methods are indexed but excluded by default in searches")
        logger.info("(This may take a while depending on API rate limits)")
        
        # Use larger batch size for better API efficiency
        # LangChain will handle internal batching to OpenAI API
        batch_size = 100
        logger.info(f"Embedding batch size: {batch_size}")
        
        with tqdm(total=len(methods), desc="Embedding methods", unit="method") as pbar:
            for i in range(0, len(methods), batch_size):
                batch = methods[i:i + batch_size]
                # Add batch - LangChain handles API batching internally
                vector_store.add_methods(batch)
                pbar.update(len(batch))
        
        # Get statistics
        stats = vector_store.get_statistics()
        logger.info(f"  ✓ Total vectors: {stats['total_vectors']}")
    
    def _get_existing_vector_ids(self, vector_store: VectorStore) -> set:
        """
        Get set of existing IDs in the vector store.
        
        Args:
            vector_store: VectorStore instance
        
        Returns:
            Set of existing IDs
        """
        try:
            # Get all IDs from the collection
            collection = vector_store.vector_store._collection
            result = collection.get(include=[])  # Just get IDs, no embeddings
            existing_ids = set(result['ids']) if result and 'ids' in result else set()
            logger.debug(f"Found {len(existing_ids)} existing vectors")
            return existing_ids
        except Exception as e:
            logger.warning(f"Could not retrieve existing IDs: {e}")
            return set()
    
    def _store_fields_in_sqlite(self, fields: List[FieldInfo], rebuild: bool):
        """
        Store fields in SQLite database.
        
        Args:
            fields: List of FieldInfo objects
            rebuild: Whether to clear existing data (already handled in _store_in_sqlite)
        """
        with SQLiteStorage(self.config.sqlite_path) as storage:
            logger.info(f"Inserting {len(fields)} fields into SQLite...")
            storage.insert_fields_batch(fields, batch_size=100)
            
            # Get statistics (just for fields)
            stats = storage.get_statistics()
            logger.info(f"  ✓ Production fields: {stats['production_fields']}")
            logger.info(f"  ✓ Test fields: {stats['test_fields']}")
    
    def _build_field_vector_index(self, fields: List[FieldInfo], rebuild: bool):
        """
        Build vector index for fields using ChromaDB and OpenAI embeddings.
        
        Args:
            fields: List of FieldInfo objects
            rebuild: Whether to clear existing data
        """
        vector_store = VectorStore(
            chroma_path=self.config.chroma_path,
            collection_name="java_fields",  # Separate collection for fields
            openai_api_key=self.config.openai_api_key,
            openai_base_url=self.config.openai_base_url
        )
        
        if rebuild:
            logger.info("Clearing existing field vector store data...")
            vector_store.clear_all()
        else:
            logger.info("Incremental mode: keeping existing field embeddings, adding new ones")
            # Filter out fields that might already be embedded
            existing_ids = self._get_existing_vector_ids(vector_store)
            if existing_ids:
                original_count = len(fields)
                fields = [f for f in fields if f.id not in existing_ids]
                logger.info(f"Filtered {original_count - len(fields)} already-embedded fields")
        
        if not fields:
            logger.info("No new fields to embed (all already exist)")
            stats = vector_store.get_statistics()
            logger.info(f"  ✓ Total field vectors: {stats['total_vectors']}")
            return
        
        # Count test vs production fields
        test_count = sum(1 for f in fields if f.is_test)
        prod_count = len(fields) - test_count
        
        # Add fields with progress bar
        logger.info(f"Creating embeddings for {len(fields)} fields...")
        logger.info(f"  - Production: {prod_count}, Test: {test_count}")
        logger.info("  Note: Test fields are indexed but excluded by default in searches")
        logger.info("(This may take a while depending on API rate limits)")
        
        # Use larger batch size for better API efficiency
        batch_size = 100
        logger.info(f"Embedding batch size: {batch_size}")
        
        with tqdm(total=len(fields), desc="Embedding fields", unit="field") as pbar:
            for i in range(0, len(fields), batch_size):
                batch = fields[i:i + batch_size]
                # Add batch - LangChain handles API batching internally
                vector_store.add_fields(batch)
                pbar.update(len(batch))
        
        # Get statistics
        stats = vector_store.get_statistics()
        logger.info(f"  ✓ Total field vectors: {stats['total_vectors']}")
    
    def _display_statistics(self, elapsed_time: float, skip_vector: bool = False):
        """
        Display final statistics about the index.
        
        Args:
            elapsed_time: Time taken to build the index (seconds)
            skip_vector: Whether vector indexing was skipped
        """
        logger.info("=" * 70)
        logger.info("INDEX STATISTICS")
        logger.info("=" * 70)
        
        # SQLite statistics
        with SQLiteStorage(self.config.sqlite_path) as storage:
            stats = storage.get_statistics()
            
            logger.info(f"Total methods indexed: {stats['total_methods']}")
            logger.info(f"  - Production: {stats['production_methods']}")
            logger.info(f"  - Test: {stats['test_methods']}")
            logger.info(f"Total fields indexed: {stats['total_fields']}")
            logger.info(f"  - Production: {stats['production_fields']}")
            logger.info(f"  - Test: {stats['test_fields']}")
            logger.info(f"Modules: {stats['modules']}")
            logger.info(f"Files processed: {stats['files']}")
            logger.info(f"Classes: {stats['classes']}")
        
        # File sizes
        sqlite_size_mb = self.config.sqlite_path.stat().st_size / (1024 * 1024)
        logger.info(f"SQLite database size: {sqlite_size_mb:.2f} MB")
        
        # ChromaDB statistics (only if vector indexing was done)
        if not skip_vector and self.config.chroma_path.exists():
            chroma_size_mb = sum(
                f.stat().st_size for f in self.config.chroma_path.rglob('*') if f.is_file()
            ) / (1024 * 1024)
            logger.info(f"ChromaDB size: {chroma_size_mb:.2f} MB")
        elif skip_vector:
            logger.info(f"ChromaDB: Skipped (--skip-vector enabled)")
        
        # Time taken
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        logger.info(f"Time taken: {minutes}m {seconds}s")
        logger.info("")

