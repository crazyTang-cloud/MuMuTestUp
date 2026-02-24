"""
Java code parser using tree-sitter.
Extracts methods, signatures, javadoc, and other metadata.
"""
import logging
from pathlib import Path
from typing import List, Optional
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

from .models import MethodInfo, FieldInfo

logger = logging.getLogger(__name__)

# Initialize tree-sitter Java language
JAVA_LANGUAGE = Language(tsjava.language())


class JavaParser:
    """
    Parses Java source files using tree-sitter.
    Extracts classes, methods, javadoc, and other metadata.
    """
    
    def __init__(self):
        self.parser = Parser(JAVA_LANGUAGE)
    
    def parse_file(
        self,
        file_path: Path,
        module_name: str,
        relative_path: str,
        is_test: bool
    ) -> tuple[List[MethodInfo], List[FieldInfo]]:
        """
        Parse a Java file and extract all methods and fields.
        
        Args:
            file_path: Absolute path to the Java file
            module_name: Maven module name
            relative_path: Path relative to project root
            is_test: Whether this is a test file
            
        Returns:
            Tuple of (methods, fields) - Lists of MethodInfo and FieldInfo objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        # CRITICAL: tree-sitter uses byte offsets, not character offsets!
        # We must keep both the string (for line-based operations) and bytes (for node extraction)
        source_bytes = bytes(source_code, 'utf-8')
        
        # Parse the source code
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node
        
        # Extract package name
        package_name = self._extract_package(root_node, source_bytes)
        
        # Extract imports (now stored with each method for LLM context)
        imports = self._extract_imports(root_node, source_bytes)
        
        # Extract all methods and fields from all classes
        methods = []
        fields = []
        class_nodes = self._find_all_classes(root_node)
        
        for class_node in class_nodes:
            class_name = self._get_class_name(class_node, source_bytes)
            if class_name:
                # Build fully qualified class name
                full_class_name = f"{package_name}.{class_name}" if package_name else class_name
                
                # Extract methods from this class
                class_methods = self._extract_methods_from_class(
                    class_node,
                    source_bytes,
                    module_name,
                    relative_path,
                    full_class_name,
                    is_test,
                    imports  # Pass imports to each method
                )
                methods.extend(class_methods)
                
                # Extract fields from this class
                class_fields = self._extract_fields_from_class(
                    class_node,
                    source_bytes,
                    module_name,
                    relative_path,
                    full_class_name,
                    is_test,
                    imports
                )
                fields.extend(class_fields)
        
        logger.debug(f"Extracted {len(methods)} methods and {len(fields)} fields from {file_path}")
        return methods, fields
    
    def _extract_package(self, root_node: Node, source_bytes: bytes) -> str:
        """Extract package declaration from the AST."""
        package_node = self._find_node_by_type(root_node, 'package_declaration')
        if package_node:
            # Get the scoped_identifier or identifier
            for child in package_node.children:
                if child.type in ('scoped_identifier', 'identifier'):
                    return self._get_text(child, source_bytes)
        return ""
    
    def _extract_imports(self, root_node: Node, source_bytes: bytes) -> List[str]:
        """Extract all import statements."""
        imports = []
        for child in root_node.children:
            if child.type == 'import_declaration':
                import_text = self._get_text(child, source_bytes)
                imports.append(import_text)
        return imports
    
    def _find_all_classes(self, root_node: Node) -> List[Node]:
        """Find all class declarations (including nested classes)."""
        classes = []
        
        def traverse(node: Node):
            if node.type in ('class_declaration', 'interface_declaration', 'enum_declaration'):
                classes.append(node)
                # Also look for nested classes
                for child in node.children:
                    if child.type == 'class_body':
                        traverse(child)
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(root_node)
        return classes
    
    def _get_class_name(self, class_node: Node, source_bytes: bytes) -> Optional[str]:
        """
        Extract class name from class declaration.
        
        Strategy: The class name is the identifier that comes immediately after
        the class/interface/enum keyword in the AST structure.
        
        Typical AST structure:
        class_declaration
        ├── modifiers (optional)
        ├── "class" keyword
        ├── identifier (THIS is the class name!)
        ├── type_parameters (optional, for generics)
        ├── superclass (optional)
        └── class_body
        """
        found_keyword = False
        
        for child in class_node.children:
            # Check if this is the class/interface/enum keyword
            # tree-sitter represents keywords as nodes with specific text
            if child.type in ('class', 'interface', 'enum'):
                found_keyword = True
                continue
            
            # Some versions use byte strings for node text
            if hasattr(child, 'text') and child.text in (b'class', b'interface', b'enum'):
                found_keyword = True
                continue
            
            # After finding the keyword, the next identifier is the class name
            if found_keyword and child.type == 'identifier':
                return self._get_text(child, source_bytes)
        
        # Fallback: if we didn't find the keyword pattern, just get first identifier
        # This shouldn't happen in well-formed Java code, but provides a safety net
        for child in class_node.children:
            if child.type == 'identifier':
                return self._get_text(child, source_bytes)
        
        return None
    
    def _extract_methods_from_class(
        self,
        class_node: Node,
        source_bytes: bytes,
        module_name: str,
        relative_path: str,
        class_name: str,
        is_test: bool,
        imports: List[str]
    ) -> List[MethodInfo]:
        """Extract all methods from a class."""
        methods = []
        
        # Find class body
        class_body = None
        for child in class_node.children:
            if child.type in ('class_body', 'interface_body', 'enum_body'):
                class_body = child
                break
        
        if not class_body:
            return methods
        
        # Iterate through class body to find methods
        for member in class_body.children:
            if member.type == 'method_declaration' or member.type == 'constructor_declaration':
                method_info = self._parse_method(
                    member,
                    source_bytes,
                    module_name,
                    relative_path,
                    class_name,
                    is_test,
                    imports
                )
                if method_info:
                    methods.append(method_info)
        
        return methods
    
    def _parse_method(
        self,
        method_node: Node,
        source_bytes: bytes,
        module_name: str,
        relative_path: str,
        class_name: str,
        is_test: bool,
        imports: List[str]
    ) -> Optional[MethodInfo]:
        """Parse a single method declaration."""
        try:
            # Get method name
            method_name = self._get_method_name(method_node, source_bytes)
            if not method_name:
                return None
            
            # Get signature (full declaration without body)
            signature = self._build_signature(method_node, source_bytes)
            
            # Get javadoc (comment before method)
            javadoc = self._get_javadoc(method_node, source_bytes)
            
            # Get method body
            body = self._get_text(method_node, source_bytes)
            
            # Get line numbers (convert to 0-based)
            start_line = method_node.start_point[0]
            end_line = method_node.end_point[0]
            
            # Generate unique ID
            method_id = MethodInfo.generate_id(module_name, relative_path, signature)
            
            return MethodInfo(
                id=method_id,
                module_name=module_name,
                file_path=relative_path,
                class_name=class_name,
                method_name=method_name,
                signature=signature,
                javadoc=javadoc,
                body=body,
                start_line=start_line,
                end_line=end_line,
                is_test=is_test,
                imports=imports if imports else None
            )
        except Exception as e:
            logger.error(f"Failed to parse method in {relative_path}: {e}")
            return None
    
    def _get_method_name(self, method_node: Node, source_bytes: bytes) -> Optional[str]:
        """Extract method name from method declaration."""
        for child in method_node.children:
            if child.type == 'identifier':
                return self._get_text(child, source_bytes)
        return None
    
    def _build_signature(self, method_node: Node, source_bytes: bytes) -> str:
        """Build method signature (modifiers + return type + name + parameters + throws)."""
        parts = []
        
        # Extract modifiers, return type, name, parameters, and throws clause
        for child in method_node.children:
            if child.type == 'modifiers':
                parts.append(self._get_text(child, source_bytes))
            elif child.type in ('type_identifier', 'void_type', 'generic_type', 
                               'array_type', 'integral_type', 'floating_point_type',
                               'boolean_type'):
                parts.append(self._get_text(child, source_bytes))
            elif child.type == 'identifier':
                parts.append(self._get_text(child, source_bytes))
            elif child.type == 'formal_parameters':
                parts.append(self._get_text(child, source_bytes))
            elif child.type == 'throws':
                # Include throws clause in signature
                parts.append(self._get_text(child, source_bytes))
        
        signature = ' '.join(parts)
        return signature.strip()
    
    def _get_javadoc(self, method_node: Node, source_bytes: bytes) -> Optional[str]:
        """
        Extract Javadoc comment before method using text-based scanning.

        设计目标：
        - 识别紧挨在方法/构造函数前面的 Javadoc 块（/** ... */）
        - 允许中间存在空行和注解（如 @Override）
        - 不依赖 tree-sitter 对 comment 的 AST 支持（comment 通常是 extras）
        """
        # Decode bytes to string for line-based operations
        source_code = source_bytes.decode('utf-8')
        # 将源码按行切分（0-based 行号与 tree-sitter 的 start_point[0] 对齐）
        lines = source_code.splitlines()
        if not lines:
            return None

        start_line = method_node.start_point[0]  # 0-based
        # 方法前面至少要有一行才可能有 Javadoc
        if start_line <= 0:
            return None

        i = start_line - 1

        # 1. 从方法定义向上，先跳过空行
        while i >= 0 and lines[i].strip() == "":
            i -= 1

        if i < 0:
            return None

        # 2. 再跳过紧邻在方法前的注解行（例如 @Override, @Test）
        #    并继续跳过它们之间的空行
        def is_annotation_line(line: str) -> bool:
            stripped = line.lstrip()
            return stripped.startswith("@")

        while i >= 0 and is_annotation_line(lines[i]):
            i -= 1
            # 注解上面可能还有空行，继续跳过
            while i >= 0 and lines[i].strip() == "":
                i -= 1

        if i < 0:
            return None

        # 3. 现在 i 指向“注解块 / 空行块”上面的第一行。
        #    如果这行不是以 */ 结尾，则认为没有紧邻的 Javadoc 块。
        if not lines[i].strip().endswith("*/"):
            return None

        end = i

        # 4. 向上寻找包含 '/**' 的起始行
        while i >= 0 and "/**" not in lines[i]:
            i -= 1

        if i < 0:
            return None

        start = i
        block = "\n".join(lines[start : end + 1]).strip()

        # 必须以 '/**' 开头才能认定为 Javadoc
        if not block.lstrip().startswith("/**"):
            return None

        return block
    
    def _get_text(self, node: Node, source_bytes: bytes) -> str:
        """
        Extract text from a node using byte offsets.
        
        CRITICAL: tree-sitter uses UTF-8 byte offsets, not character offsets!
        When source contains multi-byte characters (e.g., Chinese), we must:
        1. Slice the UTF-8 byte array using byte offsets
        2. Decode the result back to a string
        """
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8')
    
    def _find_node_by_type(self, root: Node, node_type: str) -> Optional[Node]:
        """Find first node of given type."""
        if root.type == node_type:
            return root
        
        for child in root.children:
            result = self._find_node_by_type(child, node_type)
            if result:
                return result
        
        return None
    
    def _extract_fields_from_class(
        self,
        class_node: Node,
        source_bytes: bytes,
        module_name: str,
        relative_path: str,
        class_name: str,
        is_test: bool,
        imports: List[str]
    ) -> List[FieldInfo]:
        """Extract all fields from a class."""
        fields = []
        
        # Find class body
        class_body = None
        for child in class_node.children:
            if child.type in ('class_body', 'interface_body', 'enum_body'):
                class_body = child
                break
        
        if not class_body:
            return fields
        
        # Iterate through class body to find field declarations
        for member in class_body.children:
            if member.type == 'field_declaration':
                field_infos = self._parse_field(
                    member,
                    source_bytes,
                    module_name,
                    relative_path,
                    class_name,
                    is_test,
                    imports
                )
                fields.extend(field_infos)  # Can be multiple fields in one declaration
        
        return fields
    
    def _parse_field(
        self,
        field_node: Node,
        source_bytes: bytes,
        module_name: str,
        relative_path: str,
        class_name: str,
        is_test: bool,
        imports: List[str]
    ) -> List[FieldInfo]:
        """
        Parse a field declaration (can contain multiple fields).
        e.g., private int x, y, z;
        """
        fields = []
        
        try:
            # Extract modifiers (public, private, static, final, etc.)
            modifiers = ""
            field_type = ""
            
            for child in field_node.children:
                if child.type == 'modifiers':
                    modifiers = self._get_text(child, source_bytes).strip()
                elif child.type in ('type_identifier', 'generic_type', 'array_type', 
                                   'integral_type', 'floating_point_type', 'boolean_type'):
                    field_type = self._get_text(child, source_bytes).strip()
                elif child.type == 'variable_declarator':
                    # Each variable_declarator is one field
                    field_name = None
                    initializer = None
                    
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            field_name = self._get_text(subchild, source_bytes)
                        elif subchild.type != '=' and subchild.type != 'identifier':
                            # Any non-identifier, non-equals child is part of the initializer
                            initializer = self._get_text(subchild, source_bytes).strip()
                    
                    if field_name:
                        # Get javadoc (similar to method javadoc extraction)
                        javadoc = self._get_javadoc(field_node, source_bytes)
                        
                        # Line numbers
                        start_line = field_node.start_point[0]
                        end_line = field_node.end_point[0]
                        
                        # Generate signature for ID
                        signature = f"{modifiers} {field_type} {field_name}".strip()
                        field_id = FieldInfo.generate_id(module_name, relative_path, signature)
                        
                        field_info = FieldInfo(
                            id=field_id,
                            module_name=module_name,
                            file_path=relative_path,
                            class_name=class_name,
                            field_name=field_name,
                            field_type=field_type,
                            modifiers=modifiers,
                            initializer=initializer,
                            javadoc=javadoc,
                            start_line=start_line,
                            end_line=end_line,
                            is_test=is_test,
                            imports=imports if imports else None
                        )
                        fields.append(field_info)
        
        except Exception as e:
            logger.error(f"Failed to parse field in {relative_path}: {e}")
        
        return fields

