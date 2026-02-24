"""
Data models for code indexing.
"""
from dataclasses import dataclass
from typing import Optional, List
import hashlib
import json


@dataclass
class MethodInfo:
    """
    Represents a parsed Java method with all metadata.
    """
    
    # Identity
    id: str
    
    # Location information
    module_name: str
    file_path: str  # Relative to project root
    class_name: str  # Fully qualified class name
    
    # Method details
    method_name: str
    signature: str  # Full signature with modifiers, return type, params
    javadoc: Optional[str]
    body: str  # Complete method source code
    
    # Line numbers (0-based)
    start_line: int
    end_line: int
    
    # Classification
    is_test: bool
    
    # File-level imports (same for all methods in the file)
    imports: Optional[List[str]] = None
    
    @staticmethod
    def generate_id(module_name: str, file_path: str, signature: str) -> str:
        """
        Generate unique ID for a method.
        Format: module_name:relative_path#signature_hash
        """
        # Create a hash of the signature to keep ID manageable
        sig_hash = hashlib.md5(signature.encode('utf-8')).hexdigest()[:12]
        return f"{module_name}:{file_path}#{sig_hash}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'id': self.id,
            'module_name': self.module_name,
            'file_path': self.file_path,
            'class_name': self.class_name,
            'method_name': self.method_name,
            'signature': self.signature,
            'javadoc': self.javadoc,
            'body': self.body,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'is_test': self.is_test,
            'imports': json.dumps(self.imports) if self.imports else None
        }
    
    def to_embedding_text(self) -> str:
        """
        Convert to text for vector embedding.
        Format: Class: {class_name}\nMethod: {signature}\nDoc: {javadoc}
        """
        doc_text = self.javadoc if self.javadoc else "(no documentation)"
        return f"Class: {self.class_name}\nMethod: {self.signature}\nDoc: {doc_text}"
    
    def to_metadata(self) -> dict:
        """
        Convert to metadata for ChromaDB.
        """
        return {
            'sqlite_id': self.id,
            'class_name': self.class_name,
            'module_name': self.module_name,
            'method_name': self.method_name,
            'file_path': self.file_path,
            'is_test': str(self.is_test)
        }


@dataclass
class FieldInfo:
    """
    Represents a parsed Java field (member variable) with all metadata.
    """
    
    # Identity
    id: str
    
    # Location information
    module_name: str
    file_path: str  # Relative to project root
    class_name: str  # Fully qualified class name
    
    # Field details
    field_name: str
    field_type: str  # e.g., "String", "List<Integer>", etc.
    modifiers: str  # e.g., "private static final"
    initializer: Optional[str]  # Initial value if any
    javadoc: Optional[str]
    
    # Line numbers (0-based)
    start_line: int
    end_line: int
    
    # Classification
    is_test: bool
    
    # File-level imports
    imports: Optional[List[str]] = None
    
    @staticmethod
    def generate_id(module_name: str, file_path: str, field_signature: str) -> str:
        """
        Generate unique ID for a field.
        Format: module_name:relative_path#field_signature_hash
        """
        sig_hash = hashlib.md5(field_signature.encode('utf-8')).hexdigest()[:12]
        return f"{module_name}:{file_path}#field_{sig_hash}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'id': self.id,
            'module_name': self.module_name,
            'file_path': self.file_path,
            'class_name': self.class_name,
            'field_name': self.field_name,
            'field_type': self.field_type,
            'modifiers': self.modifiers,
            'initializer': self.initializer,
            'javadoc': self.javadoc,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'is_test': self.is_test,
            'imports': json.dumps(self.imports) if self.imports else None
        }
    
    def to_embedding_text(self) -> str:
        """
        Convert to text for vector embedding.
        Format: Class: {class_name}\nField: {modifiers} {field_type} {field_name}\nDoc: {javadoc}
        """
        doc_text = self.javadoc if self.javadoc else "(no documentation)"
        init_text = f" = {self.initializer}" if self.initializer else ""
        return f"Class: {self.class_name}\nField: {self.modifiers} {self.field_type} {self.field_name}{init_text}\nDoc: {doc_text}"
    
    def to_metadata(self) -> dict:
        """
        Convert to metadata for ChromaDB.
        """
        return {
            'sqlite_id': self.id,
            'class_name': self.class_name,
            'module_name': self.module_name,
            'field_name': self.field_name,
            'field_type': self.field_type,
            'file_path': self.file_path,
            'is_test': str(self.is_test)
        }
