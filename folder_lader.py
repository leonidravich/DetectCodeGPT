#!/usr/bin/env python3
"""
Folder Lader - Repository Function and Method Scraper

This script recursively scans a repository for Python files, extracts functions and methods
using AST parsing, and stores them in DuckDB with comprehensive metadata.
"""

import os
import ast
import duckdb
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Data class to store function/method information."""
    name: str
    file_path: str
    line_number: int
    end_line: int
    function_type: str  # 'function', 'method', 'class_method', 'staticmethod'
    class_name: Optional[str]
    docstring: Optional[str]
    signature: str
    source_code: str
    decorators: List[str]
    arguments: List[str]
    return_annotation: Optional[str]
    is_async: bool
    is_generator: bool
    file_hash: str
    function_hash: str


class CodeExtractor:
    """Extracts functions and methods from Python files using AST parsing."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.functions = []
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of file content."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def get_function_hash(self, source_code: str) -> str:
        """Generate SHA256 hash of function source code."""
        return hashlib.sha256(source_code.encode('utf-8')).hexdigest()
    
    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from AST node."""
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                return node.body[0].value.value.strip()
        return None
    
    def get_decorators(self, node: ast.AST) -> List[str]:
        """Extract decorator names from AST node."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    # Handle nested attributes like module.submodule.function
                    attr_parts = []
                    current = decorator
                    while isinstance(current, ast.Attribute):
                        attr_parts.insert(0, current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        attr_parts.insert(0, current.id)
                    else:
                        # If we can't resolve the full path, just use the attribute name
                        attr_parts = [decorator.attr]
                    decorators.append(".".join(attr_parts))
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        decorators.append(decorator.func.id)
                    elif isinstance(decorator.func, ast.Attribute):
                        # Handle nested attributes in function calls
                        attr_parts = []
                        current = decorator.func
                        while isinstance(current, ast.Attribute):
                            attr_parts.insert(0, current.attr)
                            current = current.value
                        if isinstance(current, ast.Name):
                            attr_parts.insert(0, current.id)
                        else:
                            attr_parts = [decorator.func.attr]
                        decorators.append(".".join(attr_parts))
            except Exception as e:
                # If we can't parse the decorator, skip it
                logger.debug(f"Could not parse decorator: {e}")
                continue
        return decorators
    
    def get_arguments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract function arguments."""
        args = []
        
        # Positional arguments
        for arg in node.args.args:
            args.append(arg.arg)
        
        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            args.append(f"*{arg.arg}")
        
        # Varargs
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # Kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return args
    
    def get_return_annotation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def get_source_code(self, file_path: str, start_line: int, end_line: int) -> str:
        """Extract source code for a specific line range."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return ''.join(lines[start_line-1:end_line])
    
    def visit_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                      file_path: str, class_name: Optional[str] = None) -> FunctionInfo:
        """Extract information from a function or method."""
        # Determine function type
        if class_name:
            decorators = self.get_decorators(node)
            if 'classmethod' in decorators:
                function_type = 'class_method'
            elif 'staticmethod' in decorators:
                function_type = 'staticmethod'
            else:
                function_type = 'method'
        else:
            function_type = 'function'
        
        # Extract information
        docstring = self.extract_docstring(node)
        decorators = self.get_decorators(node)
        arguments = self.get_arguments(node)
        return_annotation = self.get_return_annotation(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)
        is_generator = isinstance(node, ast.FunctionDef) and node.returns is None
        
        # Get source code
        source_code = self.get_source_code(file_path, node.lineno, node.end_lineno)
        
        # Generate hashes
        file_hash = self.get_file_hash(file_path)
        function_hash = self.get_function_hash(source_code)
        
        return FunctionInfo(
            name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            end_line=node.end_lineno,
            function_type=function_type,
            class_name=class_name,
            docstring=docstring,
            signature=ast.unparse(node),
            source_code=source_code,
            decorators=decorators,
            arguments=arguments,
            return_annotation=return_annotation,
            is_async=is_async,
            is_generator=is_generator,
            file_hash=file_hash,
            function_hash=function_hash
        )
    
    def visit_class(self, node: ast.ClassDef, file_path: str):
        """Visit class and extract its methods."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    function_info = self.visit_function(item, file_path, node.name)
                    self.functions.append(function_info)
                except Exception as e:
                    logger.warning(f"Error processing method {item.name} in class {node.name} in {file_path}: {e}")
                    continue
    
    def visit_file(self, file_path: str):
        """Parse a Python file and extract all functions and methods."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if function is inside a class
                    parent_class = None
                    for parent in ast.walk(tree):
                        if (isinstance(parent, ast.ClassDef) and 
                            node in parent.body):
                            parent_class = parent.name
                            break
                    
                    if not parent_class:  # Top-level function
                        try:
                            function_info = self.visit_function(node, file_path)
                            self.functions.append(function_info)
                        except Exception as e:
                            logger.warning(f"Error processing function {node.name} in {file_path}: {e}")
                            continue
                
                elif isinstance(node, ast.ClassDef):
                    try:
                        self.visit_class(node, file_path)
                    except Exception as e:
                        logger.warning(f"Error processing class {node.name} in {file_path}: {e}")
                        continue
                    
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
    
    def scan_repository(self) -> List[FunctionInfo]:
        """Recursively scan the repository for Python files and extract functions."""
        logger.info(f"Scanning repository: {self.repo_path}")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common directories to ignore
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(python_files)} Python files")
        
        # Extract functions from each file
        for file_path in python_files:
            logger.info(f"Processing: {file_path}")
            self.visit_file(file_path)
        
        logger.info(f"Extracted {len(self.functions)} functions/methods")
        return self.functions


class DuckDBManager:
    """Manages DuckDB operations for storing function information."""
    
    def __init__(self, db_path: str = "functions.db", drop_existing: bool = False):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.create_tables(drop_existing)
    
    def create_tables(self, drop_existing=False):
        """Create the necessary tables if they don't exist."""
        if drop_existing:
            # Drop table and sequence if they exist to ensure schema is correct
            self.conn.execute("DROP TABLE IF EXISTS functions")
            self.conn.execute("DROP SEQUENCE IF EXISTS functions_id_seq")
        
        # Create sequence for auto-incrementing IDs
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS functions_id_seq")
        
        # Main functions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY DEFAULT nextval('functions_id_seq'),
                name VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                line_number INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                function_type VARCHAR NOT NULL,
                class_name VARCHAR,
                docstring TEXT,
                signature TEXT NOT NULL,
                source_code TEXT NOT NULL,
                decorators TEXT,
                arguments TEXT,
                return_annotation VARCHAR,
                is_async BOOLEAN NOT NULL,
                is_generator BOOLEAN NOT NULL,
                file_hash VARCHAR NOT NULL,
                function_hash VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # File metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path VARCHAR PRIMARY KEY,
                file_hash VARCHAR NOT NULL,
                last_modified TIMESTAMP,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_functions_file_path ON functions(file_path)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_functions_type ON functions(function_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_functions_hash ON functions(function_hash)")
        
        logger.info("Database tables created successfully")
    
    def insert_functions(self, functions: List[FunctionInfo]):
        """Insert functions into the database."""
        if not functions:
            logger.warning("No functions to insert")
            return
        
        # Prepare data for insertion
        data = []
        for func in functions:
            # Convert lists to JSON strings for storage
            decorators_str = json.dumps(func.decorators) if func.decorators else None
            arguments_str = json.dumps(func.arguments) if func.arguments else None
            
            data.append((
                func.name,
                func.file_path,
                func.line_number,
                func.end_line,
                func.function_type,
                func.class_name,
                func.docstring,
                func.signature,
                func.source_code,
                decorators_str,
                arguments_str,
                func.return_annotation,
                func.is_async,
                func.is_generator,
                func.file_hash,
                func.function_hash
            ))
        
        # Insert data
        self.conn.executemany("""
            INSERT INTO functions (
                name, file_path, line_number, end_line, function_type, class_name,
                docstring, signature, source_code, decorators, arguments, return_annotation,
                is_async, is_generator, file_hash, function_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        logger.info(f"Inserted {len(functions)} functions into database")
    
    def get_function_stats(self) -> Dict[str, Any]:
        """Get statistics about stored functions."""
        stats = {}
        
        # Total functions
        result = self.conn.execute("SELECT COUNT(*) FROM functions").fetchone()
        stats['total_functions'] = result[0] if result else 0
        
        # Functions by type
        result = self.conn.execute("""
            SELECT function_type, COUNT(*) 
            FROM functions 
            GROUP BY function_type
        """).fetchall()
        stats['by_type'] = dict(result)
        
        # Functions by file
        result = self.conn.execute("""
            SELECT file_path, COUNT(*) 
            FROM functions 
            GROUP BY file_path 
            ORDER BY COUNT(*) DESC 
            LIMIT 10
        """).fetchall()
        stats['top_files'] = dict(result)
        
        # Unique files
        result = self.conn.execute("SELECT COUNT(DISTINCT file_path) FROM functions").fetchone()
        stats['unique_files'] = result[0] if result else 0
        
        return stats
    
    def search_functions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search functions by name or content."""
        result = self.conn.execute("""
            SELECT name, file_path, line_number, function_type, class_name, docstring
            FROM functions 
            WHERE name ILIKE ? OR docstring ILIKE ? OR source_code ILIKE ?
            ORDER BY name
            LIMIT ?
        """, [f"%{query}%", f"%{query}%", f"%{query}%", limit]).fetchall()
        
        return [
            {
                'name': row[0],
                'file_path': row[1],
                'line_number': row[2],
                'function_type': row[3],
                'class_name': row[4],
                'docstring': row[5]
            }
            for row in result
        ]
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


def main():
    """Main function to run the folder lader."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Scrape functions and methods from repository")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {args.config} not found")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return
    
    # Extract configuration values
    db_path = config.get('database', {}).get('path', 'functions.db')
    drop_existing = config.get('database', {}).get('drop_existing', False)
    repo_path = config.get('repository', {}).get('path', '.')
    search = config.get('repository', {}).get('search')
    stats = config.get('repository', {}).get('stats', False)
    
    if search or stats:
        # Just query the database
        db = DuckDBManager(db_path, drop_existing=False)
        
        if stats:
            stats_data = db.get_function_stats()
            print("\n=== Function Statistics ===")
            print(f"Total functions: {stats_data['total_functions']}")
            print(f"Unique files: {stats_data['unique_files']}")
            print(f"\nFunctions by type:")
            for func_type, count in stats_data['by_type'].items():
                print(f"  {func_type}: {count}")
            print(f"\nTop files by function count:")
            for file_path, count in stats_data['top_files'].items():
                print(f"  {file_path}: {count}")
        
        if search:
            results = db.search_functions(search)
            print(f"\n=== Search Results for '{search}' ===")
            for result in results:
                print(f"\n{result['name']} ({result['function_type']})")
                print(f"  File: {result['file_path']}:{result['line_number']}")
                if result['class_name']:
                    print(f"  Class: {result['class_name']}")
                if result['docstring']:
                    print(f"  Docstring: {result['docstring'][:100]}...")
        
        db.close()
        return
    
    # Extract functions from repository
    logger.info("Starting function extraction...")
    
    extractor = CodeExtractor(repo_path)
    functions = extractor.scan_repository()
    
    if not functions:
        logger.warning("No functions found in the repository")
        return
    
    # Store in DuckDB
    db = DuckDBManager(db_path, drop_existing=drop_existing)
    db.insert_functions(functions)
    
    # Show statistics
    stats_data = db.get_function_stats()
    print("\n=== Extraction Complete ===")
    print(f"Total functions extracted: {stats_data['total_functions']}")
    print(f"Files processed: {stats_data['unique_files']}")
    print(f"\nFunctions by type:")
    for func_type, count in stats_data['by_type'].items():
        print(f"  {func_type}: {count}")
    
    db.close()
    logger.info("Function extraction completed successfully")


if __name__ == "__main__":
    main() 