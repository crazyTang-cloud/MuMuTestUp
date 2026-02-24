"""
File system walker for Java projects.
Handles multi-module Maven projects.
"""
import logging
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class JavaFileWalker:
    """
    Walks through a Java project directory and identifies:
    - Maven modules (by presence of pom.xml)
    - Java source files
    """
    
    def __init__(self, project_root: Path, ignored_dirs: Tuple[str, ...]):
        self.project_root = project_root
        self.ignored_dirs = set(ignored_dirs)
        
    def walk(self) -> List[Tuple[Path, str]]:
        """
        Walk through the project and find all Java files.
        
        Returns:
            List of tuples: (java_file_path, module_name)
            - java_file_path: absolute path to .java file
            - module_name: Maven module name, or "root" if single-module
        """
        logger.info(f"Walking project root: {self.project_root}")
        
        # First, identify all Maven modules
        modules = self._find_modules()
        logger.info(f"Found {len(modules)} module(s): {list(modules.keys())}")
        
        # Collect all Java files
        java_files = []
        for module_name, module_path in modules.items():
            files = self._find_java_files(module_path, module_name)
            java_files.extend(files)
            logger.info(f"  Module '{module_name}': {len(files)} Java files")
        
        logger.info(f"Total Java files found: {len(java_files)}")
        return java_files
    
    def _find_modules(self) -> dict:
        """
        Find all Maven modules in the project.
        
        Returns:
            Dictionary: {module_name: module_path}
        """
        modules = {}
        
        # Check if root has pom.xml (single or multi-module project)
        root_pom = self.project_root / "pom.xml"
        if root_pom.exists():
            # Check for multi-module structure
            has_submodules = False
            for item in self.project_root.iterdir():
                if item.is_dir() and item.name not in self.ignored_dirs:
                    sub_pom = item / "pom.xml"
                    if sub_pom.exists():
                        has_submodules = True
                        modules[item.name] = item
            
            # If no submodules found, treat root as single module
            if not has_submodules:
                modules["root"] = self.project_root
        else:
            # No pom.xml at root, treat entire root as a module
            logger.warning("No pom.xml found at project root, treating as single module")
            modules["root"] = self.project_root
        
        return modules
    
    def _find_java_files(self, module_path: Path, module_name: str) -> List[Tuple[Path, str]]:
        """
        Find all .java files in a module directory.
        
        Args:
            module_path: Path to the module directory
            module_name: Name of the module
            
        Returns:
            List of tuples: (java_file_path, module_name)
        """
        java_files = []
        
        # Use os.walk() for compatibility with Python < 3.12
        for root, dirs, files in os.walk(module_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            # Collect Java files
            for file in files:
                if file.endswith('.java'):
                    java_path = Path(root) / file
                    java_files.append((java_path, module_name))
        
        return java_files
    
    def get_relative_path(self, absolute_path: Path) -> str:
        """
        Convert absolute path to relative path from project root.
        
        Args:
            absolute_path: Absolute path to a file
            
        Returns:
            Relative path as string with forward slashes
        """
        try:
            relative = absolute_path.relative_to(self.project_root)
            # Use forward slashes for consistency across platforms
            return str(relative).replace('\\', '/')
        except ValueError:
            logger.warning(f"Path {absolute_path} is not relative to {self.project_root}")
            return str(absolute_path)
    
    def is_test_file(self, file_path: Path) -> bool:
        """
        Check if a Java file is a test file.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            True if it's a test file, False otherwise
        """
        path_str = str(file_path)
        # Common patterns for test files
        return (
            '/src/test/java/' in path_str.replace('\\', '/') or
            '\\src\\test\\java\\' in path_str or
            file_path.name.endswith('Test.java') or
            file_path.name.endswith('Tests.java')
        )

