"""
SQLite storage layer for code metadata and bodies.
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional

from .models import MethodInfo, FieldInfo

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """
    Manages SQLite database for storing method information.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def connect(self):
        """Establish database connection and initialize schema."""
        logger.info(f"Connecting to SQLite database: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite connection closed")
    
    def _create_schema(self):
        """Create the methods table and indexes."""
        cursor = self.conn.cursor()
        
        # Create methods table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS methods (
                id TEXT PRIMARY KEY,
                module_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                method_name TEXT NOT NULL,
                signature TEXT NOT NULL,
                javadoc TEXT,
                body TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                is_test BOOLEAN NOT NULL,
                imports TEXT
            )
        """)
        
        # Check if imports column exists (for backward compatibility with old databases)
        cursor.execute("PRAGMA table_info(methods)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'imports' not in columns:
            logger.info("Adding 'imports' column to existing methods table")
            cursor.execute("ALTER TABLE methods ADD COLUMN imports TEXT")
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_class_name 
            ON methods(class_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path 
            ON methods(file_path)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_module_name 
            ON methods(module_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lines 
            ON methods(start_line, end_line)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_is_test 
            ON methods(is_test)
        """)
        
        # Create fields table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fields (
                id TEXT PRIMARY KEY,
                module_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                field_name TEXT NOT NULL,
                field_type TEXT NOT NULL,
                modifiers TEXT,
                initializer TEXT,
                javadoc TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                is_test BOOLEAN NOT NULL,
                imports TEXT
            )
        """)
        
        # Create indexes for fields table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fields_class_name 
            ON fields(class_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fields_file_path 
            ON fields(file_path)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fields_module_name 
            ON fields(module_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fields_is_test 
            ON fields(is_test)
        """)
        
        self.conn.commit()
        logger.info("SQLite schema initialized")
    
    def clear_all(self):
        """Clear all data from the database (for rebuilding index)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM methods")
        cursor.execute("DELETE FROM fields")
        self.conn.commit()
        logger.info("Cleared all data from database (methods and fields)")
    
    def insert_method(self, method: MethodInfo):
        """
        Insert a single method into the database.
        
        Args:
            method: MethodInfo object to insert
        """
        cursor = self.conn.cursor()
        method_dict = method.to_dict()
        cursor.execute("""
            INSERT OR REPLACE INTO methods 
            (id, module_name, file_path, class_name, method_name, 
             signature, javadoc, body, start_line, end_line, is_test, imports)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            method.id,
            method.module_name,
            method.file_path,
            method.class_name,
            method.method_name,
            method.signature,
            method.javadoc,
            method.body,
            method.start_line,
            method.end_line,
            method.is_test,
            method_dict['imports']
        ))
    
    def insert_methods_batch(self, methods: List[MethodInfo], batch_size: int = 100):
        """
        Insert multiple methods in batches for better performance.
        
        Args:
            methods: List of MethodInfo objects
            batch_size: Number of records to insert per batch
        """
        cursor = self.conn.cursor()
        
        for i in range(0, len(methods), batch_size):
            batch = methods[i:i + batch_size]
            data = [
                (
                    m.id, m.module_name, m.file_path, m.class_name, m.method_name,
                    m.signature, m.javadoc, m.body, m.start_line, m.end_line, m.is_test,
                    m.to_dict()['imports']
                )
                for m in batch
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO methods 
                (id, module_name, file_path, class_name, method_name, 
                 signature, javadoc, body, start_line, end_line, is_test, imports)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            self.conn.commit()
            logger.debug(f"Inserted batch of {len(batch)} methods")
    
    def get_method_by_id(self, method_id: str) -> Optional[dict]:
        """
        Retrieve a method by its ID.
        
        Args:
            method_id: The method ID
            
        Returns:
            Dictionary with method data, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM methods WHERE id = ?", (method_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_methods_by_class(self, class_name: str) -> List[dict]:
        """
        Retrieve all methods from a specific class.
        
        Args:
            class_name: Fully qualified class name
            
        Returns:
            List of dictionaries with method data
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM methods WHERE class_name = ?", (class_name,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_method_by_location(self, class_name: str, line_number: int) -> Optional[dict]:
        """
        Find a method by class name and line number.
        
        Args:
            class_name: Fully qualified class name
            line_number: Line number within the method
            
        Returns:
            Dictionary with method data, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM methods 
            WHERE class_name = ? 
            AND start_line <= ? 
            AND end_line >= ?
            LIMIT 1
        """, (class_name, line_number, line_number))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def insert_field(self, field: FieldInfo):
        """
        Insert a single field into the database.
        
        Args:
            field: FieldInfo object to insert
        """
        cursor = self.conn.cursor()
        field_dict = field.to_dict()
        cursor.execute("""
            INSERT OR REPLACE INTO fields 
            (id, module_name, file_path, class_name, field_name, 
             field_type, modifiers, initializer, javadoc, start_line, end_line, is_test, imports)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            field.id,
            field.module_name,
            field.file_path,
            field.class_name,
            field.field_name,
            field.field_type,
            field.modifiers,
            field.initializer,
            field.javadoc,
            field.start_line,
            field.end_line,
            field.is_test,
            field_dict['imports']
        ))
    
    def insert_fields_batch(self, fields: List[FieldInfo], batch_size: int = 100):
        """
        Insert multiple fields in batches for better performance.
        
        Args:
            fields: List of FieldInfo objects
            batch_size: Number of records to insert per batch
        """
        cursor = self.conn.cursor()
        
        for i in range(0, len(fields), batch_size):
            batch = fields[i:i + batch_size]
            data = [
                (
                    f.id, f.module_name, f.file_path, f.class_name, f.field_name,
                    f.field_type, f.modifiers, f.initializer, f.javadoc, 
                    f.start_line, f.end_line, f.is_test,
                    f.to_dict()['imports']
                )
                for f in batch
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO fields 
                (id, module_name, file_path, class_name, field_name, 
                 field_type, modifiers, initializer, javadoc, start_line, end_line, is_test, imports)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            self.conn.commit()
            logger.debug(f"Inserted batch of {len(batch)} fields")
    
    def get_field_by_id(self, field_id: str) -> Optional[dict]:
        """
        Retrieve a field by its ID.
        
        Args:
            field_id: The field ID
            
        Returns:
            Dictionary with field data, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM fields WHERE id = ?", (field_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_fields_by_class(self, class_name: str) -> List[dict]:
        """
        Retrieve all fields from a specific class.
        
        Args:
            class_name: Fully qualified class name
            
        Returns:
            List of dictionaries with field data
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM fields WHERE class_name = ?", (class_name,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        cursor = self.conn.cursor()
        
        # Total methods
        cursor.execute("SELECT COUNT(*) as total FROM methods")
        total = cursor.fetchone()['total']
        
        # Production vs test methods
        cursor.execute("SELECT COUNT(*) as count FROM methods WHERE is_test = 0")
        production = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM methods WHERE is_test = 1")
        test = cursor.fetchone()['count']
        
        # Total fields
        cursor.execute("SELECT COUNT(*) as total FROM fields")
        total_fields = cursor.fetchone()['total']
        
        # Production vs test fields
        cursor.execute("SELECT COUNT(*) as count FROM fields WHERE is_test = 0")
        production_fields = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM fields WHERE is_test = 1")
        test_fields = cursor.fetchone()['count']
        
        # Modules
        cursor.execute("SELECT COUNT(DISTINCT module_name) as count FROM methods")
        modules = cursor.fetchone()['count']
        
        # Files
        cursor.execute("SELECT COUNT(DISTINCT file_path) as count FROM methods")
        files = cursor.fetchone()['count']
        
        # Classes
        cursor.execute("SELECT COUNT(DISTINCT class_name) as count FROM methods")
        classes = cursor.fetchone()['count']
        
        return {
            'total_methods': total,
            'production_methods': production,
            'test_methods': test,
            'total_fields': total_fields,
            'production_fields': production_fields,
            'test_fields': test_fields,
            'modules': modules,
            'files': files,
            'classes': classes
        }

