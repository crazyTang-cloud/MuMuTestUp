"""
Vector store using ChromaDB with OpenAI embeddings.
"""
import logging
import shutil
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .models import MethodInfo, FieldInfo

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector embeddings for code using ChromaDB and OpenAI.
    """
    
    def __init__(
        self,
        chroma_path: Path,
        collection_name: str,
        openai_api_key: str,
        openai_base_url: str
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        # Initialize OpenAI embeddings
        logger.info(f"Initializing OpenAI embeddings (base_url: {openai_base_url})")
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        # Initialize ChromaDB client
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB at {self.chroma_path}")
        
        # Create Chroma vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_path)
        )
        
        logger.info("Vector store initialized")
    
    def clear_all(self):
        """
        Clear all data from the vector store and clean up orphaned directories.
        
        This method:
        1. Deletes the collection from ChromaDB metadata
        2. Removes orphaned UUID directories (old HNSW index data)
        3. Recreates a fresh collection
        """
        try:
            # Step 1: Delete the collection logically
            self.vector_store.delete_collection()
            logger.info("Deleted collection from ChromaDB metadata")
            
            # Step 2: Clean up orphaned UUID directories
            self._cleanup_orphaned_directories()
            
            # Step 3: Reinitialize with a fresh collection
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.chroma_path)
            )
            logger.info("Vector store cleared and reinitialized")
        except Exception as e:
            logger.warning(f"Could not clear vector store: {e}")
    
    def _cleanup_orphaned_directories(self):
        """
        Remove orphaned UUID directories from the ChromaDB persist directory.
        
        ChromaDB creates a new UUID directory for each collection's HNSW index.
        After delete_collection(), these directories become orphaned and should be cleaned up.
        
        This method safely removes directories that:
        - Are in the chroma_path
        - Look like UUIDs (contain exactly 4 hyphens, typical UUID format)
        - Are directories (not files like chroma.sqlite3)
        """
        if not self.chroma_path.exists():
            return
        
        removed_count = 0
        failed_removals = []
        
        try:
            for item in self.chroma_path.iterdir():
                # Only process directories
                if not item.is_dir():
                    continue
                
                # Check if directory name looks like a UUID (has 4 hyphens)
                # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
                if item.name.count('-') == 4 and len(item.name) == 36:
                    try:
                        shutil.rmtree(item)
                        removed_count += 1
                        logger.debug(f"Removed orphaned directory: {item.name}")
                    except Exception as e:
                        failed_removals.append((item.name, str(e)))
                        logger.debug(f"Failed to remove {item.name}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} orphaned index directory(ies)")
            
            if failed_removals:
                logger.warning(
                    f"Could not remove {len(failed_removals)} directory(ies): "
                    f"{', '.join(name for name, _ in failed_removals[:3])}"
                    + ("..." if len(failed_removals) > 3 else "")
                )
        except Exception as e:
            logger.warning(f"Error during directory cleanup: {e}")
    
    def add_methods(self, methods: List[MethodInfo]):
        """
        Add methods to the vector store in one batch.
        
        LangChain's add_texts will handle internal batching for API calls automatically,
        which is more efficient than manually splitting into small batches.
        
        Args:
            methods: List of MethodInfo objects to add
        """
        if not methods:
            return
        
        logger.debug(f"Adding {len(methods)} methods to vector store")
        
        try:
            # Prepare all data at once
            texts = [method.to_embedding_text() for method in methods]
            metadatas = [method.to_metadata() for method in methods]
            ids = [method.id for method in methods]
            
            # Add to vector store - LangChain handles API batching internally
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.debug(f"Successfully added {len(methods)} methods")
            
        except Exception as e:
            logger.error(f"Failed to add {len(methods)} methods: {e}")
            raise
    
    def add_fields(self, fields: List[FieldInfo]):
        """
        Add fields to the vector store in one batch.
        
        LangChain's add_texts will handle internal batching for API calls automatically.
        
        Args:
            fields: List of FieldInfo objects to add
        """
        if not fields:
            return
        
        logger.debug(f"Adding {len(fields)} fields to vector store")
        
        try:
            # Prepare all data at once
            texts = [field.to_embedding_text() for field in fields]
            metadatas = [field.to_metadata() for field in fields]
            ids = [field.id for field in fields]
            
            # Add to vector store - LangChain handles API batching internally
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.debug(f"Successfully added {len(fields)} fields")
            
        except Exception as e:
            logger.error(f"Failed to add {len(fields)} fields: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict = None,
        include_tests: bool = False
    ) -> List[dict]:
        """
        Search for methods using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"module_name": "user-service"})
            include_tests: If False (default), exclude test methods from results.
                          Set to True to include test methods in search results.
            
        Returns:
            List of dictionaries with method metadata and similarity scores
            
        Note:
            By default, test methods are excluded from search results because:
            - The primary use case is finding production code for test fixing
            - Test methods in results would add noise and reduce relevance
            - All test methods are still stored in SQLite for analysis
        """
        try:
            # Apply test filter if needed
            if not include_tests:
                # Merge with existing filter to exclude test methods
                if filter_dict is None:
                    filter_dict = {}
                else:
                    filter_dict = filter_dict.copy()  # Don't modify caller's dict
                filter_dict['is_test'] = 'False'
                logger.debug("Excluding test methods from search results (include_tests=False)")
            
            # Perform similarity search
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # Return all metadata fields (supports both methods and fields)
                result = dict(doc.metadata)  # Copy all metadata
                result['score'] = score
                result['content'] = doc.page_content
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> dict:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get collection
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'total_vectors': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_vectors': 0,
                'collection_name': self.collection_name
            }

