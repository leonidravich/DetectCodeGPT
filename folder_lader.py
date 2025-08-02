#!/usr/bin/env python3
"""
Folder Lader - Repository Function and Method Scraper (Optimized for Huge Repos)

This script recursively scans a repository for Python files, extracts functions and methods
using AST parsing, and stores them in DuckDB with comprehensive metadata.
OPTIMIZED VERSION: Includes parallel processing, git caching, and batch operations.
"""

import os
import ast
import duckdb
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import concurrent.futures
import threading
import time
from functools import lru_cache
from collections import defaultdict, deque
import multiprocessing as mp
import re
from abc import ABC, abstractmethod
import tree_sitter
from tree_sitter import Language

# Tree-sitter is required for C++ parsing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GitCache:
    """High-performance git operations cache for batch processing."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.file_blame_cache = {}
        self.file_changes_cache = {}
        self.commit_info_cache = {}
        self.modified_files_cache = {}
        self.new_functions_cache = {}  # Cache for new functions introduced per year
        self._lock = threading.Lock()
    
    def update_repo_path(self, new_repo_path: str):
        """Update the repository path and clear all caches since they're now invalid."""
        with self._lock:
            self.repo_path = Path(new_repo_path)
            # Clear all caches since they're based on the old repo path
            self.file_blame_cache.clear()
            self.file_changes_cache.clear()
            self.commit_info_cache.clear()
            self.modified_files_cache.clear()
            self.new_functions_cache.clear()
            logger.info(f"GitCache repo_path updated to: {new_repo_path}")
        
    def batch_get_modified_files_in_year(self, year: int, supported_extensions: List[str] = None) -> Set[str]:
        """Get all files modified in a specific year in one git operation."""
        cache_key = f"modified_files_{year}"
        if cache_key in self.modified_files_cache:
            return self.modified_files_cache[cache_key]
            
        # Default to Python files if no extensions provided (backward compatibility)
        if supported_extensions is None:
            raise ValueError("No supported extensions provided")
            
        try:
            # Build git log command with path filters for supported extensions
            log_cmd = ['git', 'log', '--name-only', '--format=', f'--since={year}-01-01', f'--until={year}-12-31']
            
            # Add path filters for each supported extension
            for ext in supported_extensions:
                log_cmd.extend(['--', f'**/*.{ext[1:]}'])  # Remove the dot from extension, support nested subfolders
            
            result = subprocess.run(
                log_cmd,
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )
            
            # Decode the output with error handling
            try:
                log_output = result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with error handling
                log_output = result.stdout.decode('utf-8', errors='ignore')
            
            modified_files = set()
            for line in log_output.strip().split('\n'):
                if line.strip() and any(line.strip().endswith(ext) for ext in supported_extensions):
                    modified_files.add(line.strip())
            
            self.modified_files_cache[cache_key] = modified_files
            logger.info(f"Cached {len(modified_files)} modified files for year {year} (extensions: {supported_extensions})")
            return modified_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting modified files for {year}: {e}")
            return set()
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error getting modified files for {year}: {e}")
            return set()
        except Exception as e:
            logger.error(f"Unexpected error getting modified files for {year}: {e}")
            return set()
    
    def batch_get_file_blame(self, file_path: str) -> Dict[int, Dict[str, str]]:
        """Get git blame for entire file in one operation."""
        if file_path in self.file_blame_cache:
            return self.file_blame_cache[file_path]
            
        try:
            result = subprocess.run(
                ['git', 'blame', '--porcelain', file_path],
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )
            
            # Decode the output with error handling
            try:
                blame_output = result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with error handling
                blame_output = result.stdout.decode('utf-8', errors='ignore')
            
            blame_data = {}
            current_line = 1
            lines = blame_output.split('\n')
            i = 0
            
            while i < len(lines):
                line = lines[i]
                if line and len(line.split()) > 0:
                    commit_hash = line.split()[0]
                    if len(commit_hash) == 40:  # Full commit hash
                        # Parse porcelain format
                        commit_info = {'commit_hash': commit_hash}
                        i += 1
                        
                        # Parse additional info
                        while i < len(lines) and not lines[i].startswith('\t'):
                            if lines[i].startswith('author '):
                                commit_info['commit_author'] = lines[i][7:]
                            elif lines[i].startswith('author-time '):
                                timestamp = int(lines[i][12:])
                                commit_info['commit_date'] = datetime.fromtimestamp(timestamp).isoformat()
                            elif lines[i].startswith('summary '):
                                commit_info['commit_message'] = lines[i][8:]
                            i += 1
                        
                        blame_data[current_line] = commit_info
                        current_line += 1
                i += 1
            
            with self._lock:
                self.file_blame_cache[file_path] = blame_data
            return blame_data
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error getting blame for {file_path}: {e}")
            return {}
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error getting blame for {file_path}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Unexpected error getting blame for {file_path}: {e}")
            return {}
    
    def get_line_commit_info(self, file_path: str, line_number: int) -> Dict[str, Any]:
        """Get commit info for a specific line using cached blame data."""
        blame_data = self.batch_get_file_blame(file_path)
        return blame_data.get(line_number, {})
    
    def batch_get_new_functions_in_year(self, year: int, supported_extensions: List[str] = None) -> Dict[str, Set[str]]:
        """Get all new functions introduced in a specific year using git diff between first and last commit."""
        cache_key = f"new_functions_{year}"
        if cache_key in self.new_functions_cache:
            return self.new_functions_cache[cache_key]
            
        if supported_extensions is None:
            raise ValueError("No supported extensions provided")
        
        logger.info(f"Getting new functions for {year} with supported extensions: {supported_extensions}")

        try:
            # Get first and last commit hashes for the year
            commits_result = subprocess.run(
                ['git', 'log', '--format=%H', f'--since={year}-01-01', f'--until={year}-12-31'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hashes = [h.strip() for h in commits_result.stdout.strip().split('\n') if h.strip()]
            
            if not commit_hashes:
                # No commits in this year
                with self._lock:
                    self.new_functions_cache[cache_key] = {}
                return {}
            
            # Get the first and last commit of the year
            first_commit = commit_hashes[-1]  # Last in chronological order (oldest)
            last_commit = commit_hashes[0]    # First in chronological order (newest)
            
            # Use git diff between first and last commit to get changes only for supported file types
            # Build git diff command with path filters for supported extensions
            diff_cmd = ['git', 'diff', '--unified=0', f'{first_commit}..{last_commit}']
            
            # Add path filters for each supported extension
            for ext in supported_extensions:
                diff_cmd.extend(['--', f'**/*.{ext[1:]}'])  # Remove the dot from extension, support nested subfolders
            
            logger.info(f"Running git diff command: {diff_cmd}")

            diff_result = subprocess.run(
                diff_cmd,
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )
            
            # Decode the output with error handling
            try:
                diff_output = diff_result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with error handling
                diff_output = diff_result.stdout.decode('utf-8', errors='ignore')
            
            new_functions = {}  # file_path -> set of function names
            current_file = None

            # Parse the unified diff output
            for line in diff_output.split('\n'):
                line = line.strip()
                
                # Check for file header (e.g., "+++ b/path/to/file")
                if line.startswith('+++ b/'):
                    current_file = line[6:]  # Remove "+++ b/" prefix
                    if not any(current_file.endswith(ext) for ext in supported_extensions):
                        current_file = None
                    continue
                
                # Check for added lines (new functions)
                if line.startswith('+') and not line.startswith('+++') and current_file:
                    function_name = self._extract_function_name_from_line(line[1:], current_file)
                    if function_name:
                        if current_file not in new_functions:
                            new_functions[current_file] = set()
                        new_functions[current_file].add(function_name)
            
            with self._lock:
                self.new_functions_cache[cache_key] = new_functions
            logger.info(f"Cached {sum(len(funcs) for funcs in new_functions.values())} new functions for year {year}")
            return new_functions
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting new functions for {year}: {e}")
            return {}
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error getting new functions for {year}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting new functions for {year}: {e}")
            return {}
    

    
    def _extract_function_name_from_line(self, line: str, file_path: str) -> Optional[str]:
        """Extract function name from a line of code."""
        line = line.strip()
        
        # Python function patterns
        if file_path.endswith('.py'):
            # Match: def function_name( or async def function_name(
            match = re.match(r'^\s*(?:async\s+)?def\s+(\w+)', line)
            if match:
                return match.group(1)
        
        # C function patterns
        elif file_path.endswith(('.c', '.h', '.cpp', '.hpp')):
            # Match: return_type function_name( or function_name(
            match = re.match(r'^\s*(?:\w+\s+)*(\w+)\s*\(', line)
            if match:
                return match.group(1)
        
        return None
    
    def is_file_modified_in_year(self, file_path: str, year: int, supported_extensions: List[str] = None) -> bool:
        """Check if file was modified in year using cached data."""
        modified_files = self.batch_get_modified_files_in_year(year, supported_extensions)
        return file_path in modified_files
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self.file_blame_cache.clear()
            self.file_changes_cache.clear()
            self.commit_info_cache.clear()
            self.modified_files_cache.clear()
            self.new_functions_cache.clear()


class PerformanceMetrics:
    """Track performance metrics for optimization analysis."""
    
    def __init__(self):
        self.start_time = time.time()
        self.file_count = 0
        self.function_count = 0
        self.git_ops_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.processing_times = deque(maxlen=1000)
        
    def record_file_processed(self, processing_time: float):
        self.file_count += 1
        self.processing_times.append(processing_time)
        
    def record_function_found(self):
        self.function_count += 1
        
    def record_git_operation(self):
        self.git_ops_count += 1
        
    def record_cache_hit(self):
        self.cache_hits += 1
        
    def record_cache_miss(self):
        self.cache_misses += 1
        
    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        avg_file_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'elapsed_time': elapsed,
            'files_processed': self.file_count,
            'functions_found': self.function_count,
            'git_operations': self.git_ops_count,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_file_processing_time': avg_file_time,
            'files_per_second': self.file_count / elapsed if elapsed > 0 else 0,
            'functions_per_second': self.function_count / elapsed if elapsed > 0 else 0
        }


@dataclass
class FunctionInfo:
    """Data class to store function/method information."""
    name: str
    file_path: str
    line_number: int
    end_line: int
    function_type: str  # 'function', 'method', 'class_method', 'staticmethod', 'constructor', 'destructor'
    class_name: Optional[str]
    docstring: Optional[str]
    signature: str
    source_code: str
    decorators: List[str]
    arguments: List[str]
    return_annotation: Optional[str]
    is_async: bool
    is_generator: bool
    language: str  # Programming language: 'python', 'c'
    namespace: Optional[str] = None  # For C namespaces (not used in C)
    template_parameters: Optional[str] = None  # For C templates (not used in C)
    # Git commit metadata
    commit_hash: Optional[str] = None
    commit_date: Optional[str] = None
    commit_author: Optional[str] = None
    commit_message: Optional[str] = None


class BaseParser(ABC):
    """Abstract base class for language-specific parsers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = self._get_language_name()
        self.extensions = self._get_file_extensions()
        self.parsing_config = config.get('languages', {}).get('parsing', {}).get(self.language, {})
    
    @abstractmethod
    def _get_language_name(self) -> str:
        """Return the language identifier."""
        pass
    
    @abstractmethod
    def _get_file_extensions(self) -> List[str]:
        """Return supported file extensions for this language."""
        pass
    
    @abstractmethod
    def extract_functions(self, file_path: str, file_content: str, git_cache: 'GitCache', 
                         target_year: Optional[int] = None) -> List[FunctionInfo]:
        """Extract functions from the given file content."""
        pass
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        return any(file_path.endswith(ext) for ext in self.extensions)
    
    def get_commit_info(self, file_path: str, line_number: int, git_cache: 'GitCache') -> Dict[str, Any]:
        """Get git commit information for a specific line."""
        try:
            relative_path = os.path.relpath(file_path, git_cache.repo_path)
            commit_info = git_cache.get_line_commit_info(relative_path, line_number)
            return commit_info if commit_info else {}
        except Exception as e:
            logger.warning(f"Error getting commit info for {file_path}:{line_number}: {e}")
            return {}


class PythonParser(BaseParser):
    """Parser for Python files using AST."""
    
    def _get_language_name(self) -> str:
        return "python"
    
    def _get_file_extensions(self) -> List[str]:
        return self.config.get('languages', {}).get('extensions', {}).get('python', ['.py'])
    
    def extract_functions(self, file_path: str, file_content: str, git_cache: 'GitCache', 
                         target_year: Optional[int] = None) -> List[FunctionInfo]:
        """Extract functions from Python file using AST."""
        functions = []
        
        try:
            tree = ast.parse(file_content)
            
            # Process top-level nodes
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_info = self._visit_function(node, file_path, file_content, git_cache, target_year=target_year)
                    if function_info:
                        functions.append(function_info)
                elif isinstance(node, ast.ClassDef):
                    class_functions = self._visit_class(node, file_path, file_content, git_cache, target_year)
                    functions.extend(class_functions)
            
            return functions
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return []
    
    def _visit_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                       file_path: str, file_content: str, git_cache: 'GitCache',
                       class_name: Optional[str] = None, target_year: Optional[int] = None) -> Optional[FunctionInfo]:
        """Extract information from a function or method."""
        # Check if function was modified in target year
        if target_year:
            try:
                was_modified_in_year = self._is_function_modified_in_year(
                    file_path, node.lineno, node.end_lineno, target_year, git_cache
                )
                if not was_modified_in_year:
                    return None
            except Exception as e:
                logger.warning(f"Error checking if function {node.name} was modified in {file_path}: {e}")
        
        # Determine function type
        decorators = self._get_decorators(node)
        if class_name:
            if 'classmethod' in decorators:
                function_type = 'class_method'
            elif 'staticmethod' in decorators:
                function_type = 'staticmethod'
            else:
                function_type = 'method'
        else:
            function_type = 'function'
        
        # Extract information
        docstring = self._extract_docstring(node) if self.parsing_config.get('extract_docstrings', True) else None
        arguments = self._get_arguments(node)
        return_annotation = self._get_return_annotation(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)
        is_generator = self._is_generator(node) if self.parsing_config.get('extract_generators', True) else False
        
        # Get source code
        source_code = self._get_source_code(file_content, node.lineno, node.end_lineno)
        
        # Get git commit information
        commit_info = self.get_commit_info(file_path, node.lineno, git_cache)
        
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
            decorators=decorators if self.parsing_config.get('extract_decorators', True) else [],
            arguments=arguments,
            return_annotation=return_annotation,
            is_async=is_async,
            is_generator=is_generator,
            language=self.language,
            namespace=None,  # Python doesn't have namespaces like C++
            template_parameters=None,  # Python doesn't have templates
            commit_hash=commit_info.get('commit_hash'),
            commit_date=commit_info.get('commit_date'),
            commit_author=commit_info.get('commit_author'),
            commit_message=commit_info.get('commit_message')
        )
    
    def _visit_class(self, node: ast.ClassDef, file_path: str, file_content: str, 
                    git_cache: 'GitCache', target_year: Optional[int] = None) -> List[FunctionInfo]:
        """Visit class and extract its methods."""
        class_functions = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    function_info = self._visit_function(item, file_path, file_content, git_cache, node.name, target_year)
                    if function_info:
                        class_functions.append(function_info)
                except Exception as e:
                    logger.warning(f"Error processing method {item.name} in class {node.name} in {file_path}: {e}")
                    continue
        return class_functions
    
    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from AST node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                return node.body[0].value.value.strip()
        return None
    
    def _get_decorators(self, node: ast.AST) -> List[str]:
        """Extract decorator names from AST node."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    attr_parts = []
                    current = decorator
                    while isinstance(current, ast.Attribute):
                        attr_parts.insert(0, current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        attr_parts.insert(0, current.id)
                    else:
                        attr_parts = [decorator.attr]
                    decorators.append(".".join(attr_parts))
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        decorators.append(decorator.func.id)
                    elif isinstance(decorator.func, ast.Attribute):
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
                logger.debug(f"Could not parse decorator: {e}")
                continue
        return decorators
    
    def _get_arguments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
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
    
    def _get_return_annotation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function is a generator."""
        for item in ast.walk(node):
            if isinstance(item, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _get_source_code(self, file_content: str, start_line: int, end_line: int) -> str:
        """Extract source code for a specific line range."""
        lines = file_content.splitlines(keepends=True)
        return ''.join(lines[start_line-1:end_line])
    
    def _is_function_modified_in_year(self, file_path: str, start_line: int, end_line: int, 
                                     year: int, git_cache: 'GitCache') -> bool:
        """Check if a function was introduced in the specified year (optimized for new functions only)."""
        relative_path = os.path.relpath(file_path, git_cache.repo_path)
        relative_path_normalized = relative_path.replace(os.sep, '/')
        
        # Get cached new functions for this year
        new_functions = git_cache.batch_get_new_functions_in_year(year, self.extensions)
        
        # Check if this file has any new functions
        if relative_path_normalized not in new_functions:
            return False
        
        # For now, if the file has new functions, we'll consider all functions in the file
        # as potentially new. This is a simplified approach for performance.
        # In a more sophisticated implementation, we could parse the function name
        # and check if it's in the new_functions set.
        return True





class CParser(BaseParser):
    """Parser for C files using Tree-sitter parsing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.parser = None
        self.language = None
        self._initialize_tree_sitter()
    
    def _initialize_tree_sitter(self):
        """Initialize Tree-sitter parser for C using pre-built package."""
        try:
            # Create parser
            self.parser = tree_sitter.Parser()
            
            # Import and use the pre-built tree-sitter-c package
            try:
                import tree_sitter_c
                
                # Create Language object using the correct pattern from working example
                self.language = Language(tree_sitter_c.language())
                self.parser.language = self.language
                logger.info("Tree-sitter C parser initialized successfully using pre-built package")
                
            except ImportError:
                logger.error("tree-sitter-c not installed")
                raise RuntimeError("tree-sitter-c not installed. Install with: pip install tree-sitter-c")
            except Exception as e:
                logger.error(f"Failed to load C language: {e}")
                raise RuntimeError(f"C language not available: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter parser: {e}")
            raise RuntimeError(f"Tree-sitter parser initialization failed: {e}")
    
    def _get_language_name(self) -> str:
        return "c"
    
    def _get_file_extensions(self) -> List[str]:
        return self.config.get('languages', {}).get('extensions', {}).get('c', ['.c', '.h'])
    
    def extract_functions(self, file_path: str, file_content: str, git_cache: 'GitCache', 
                         target_year: Optional[int] = None) -> List[FunctionInfo]:
        """Extract functions from C file using Tree-sitter parsing."""
        functions = []
        try:
            # Parse the file with Tree-sitter
            tree = self.parser.parse(file_content.encode('utf-8'))
            root_node = tree.root_node
            
            # Extract functions
            function_nodes = self._find_function_nodes_tree_sitter(root_node)
            logger.debug(f"Found {len(function_nodes)} function nodes in {file_path}")

            for func_node in function_nodes:
                try:
                    function_info = self._create_function_info_tree_sitter(
                        func_node, file_path, file_content, git_cache, target_year
                    )
                    if function_info:
                        functions.append(function_info)
                except Exception as e:
                    logger.warning(f"Error processing C function with Tree-sitter in {file_path}: {e}")
                    continue
            
            return functions
            
        except Exception as e:
            logger.error(f"Tree-sitter parsing failed for {file_path}: {e}")
            raise
    
    def _find_function_nodes_tree_sitter(self, root_node):
        """Find function definition nodes using Tree-sitter."""
        function_nodes = []
        
        def find_functions_recursive(node):
            if node.type == 'function_definition':
                function_nodes.append(node)
            for child in node.children:
                find_functions_recursive(child)
        
        find_functions_recursive(root_node)
        return function_nodes
    
    def _create_function_info_tree_sitter(self, func_node, file_path: str, 
                                    file_content: str, git_cache: 'GitCache',
                                    target_year: Optional[int] = None) -> Optional[FunctionInfo]:
        """Create FunctionInfo from a Tree-sitter node."""
        try:
            # Extract function details from Tree-sitter node
            function_name = self._extract_function_name_tree_sitter(func_node)
            if not function_name:
                return None
            # Get line numbers
            start_line = func_node.start_point[0] + 1
            end_line = func_node.end_point[0] + 1
            
            # Check if function was modified in target year
            if target_year:
                try:
                    was_modified_in_year = self._is_function_modified_in_year(
                        file_path, start_line, end_line, target_year, git_cache
                    )
                    if not was_modified_in_year:
                        return None
                except Exception as e:
                    logger.warning(f"Error checking if C function was modified: {e}")


            # Extract function details
            return_type = self._extract_return_type_tree_sitter(func_node)
            parameters = self._extract_parameters_tree_sitter(func_node)
            
            # Determine function type (C only has functions, no classes/methods)
            function_type = 'function'
            
            # Build signature
            signature = f"{return_type} {function_name}({parameters})"
            
            # Extract arguments
            arguments = self._parse_c_parameters(parameters)
            
            # Get source code
            source_code = func_node.text.decode('utf-8')
            
            # Extract documentation comment
            docstring = self._extract_c_documentation_tree_sitter(file_content, start_line)
            
            # Get git commit information
            commit_info = self.get_commit_info(file_path, start_line, git_cache)
            
            return FunctionInfo(
                name=function_name,
                file_path=str(file_path),
                line_number=start_line,
                end_line=end_line,
                function_type=function_type,
                class_name=None,  # C doesn't have classes
                docstring=docstring if self.parsing_config.get('extract_comments', True) else None,
                signature=signature,
                source_code=source_code,
                decorators=[],  # C doesn't have decorators
                arguments=arguments,
                return_annotation=return_type,
                is_async=False,  # C functions are not async
                is_generator=False,  # C doesn't have generators
                language=self._get_language_name(),
                namespace=None,  # C doesn't have namespaces like C++
                template_parameters=None,  # C doesn't have templates
                commit_hash=commit_info.get('commit_hash'),
                commit_date=commit_info.get('commit_date'),
                commit_author=commit_info.get('commit_author'),
                commit_message=commit_info.get('commit_message')
            )
            
        except Exception as e:
            logger.warning(f"Error creating function info from Tree-sitter node: {e}")
            return None
    
    def _extract_function_name_tree_sitter(self, func_node):
        """Extract function name from Tree-sitter node."""
        for child in func_node.children:
            if child.type == 'function_declarator':
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        return grandchild.text.decode('utf-8')
        return None
    
    def _extract_return_type_tree_sitter(self, func_node):
        """Extract return type from Tree-sitter node."""
        for child in func_node.children:
            if child.type in ['primitive_type', 'type_identifier']:
                return child.text.decode('utf-8')
        return "void"  # Default return type
    
    def _extract_parameters_tree_sitter(self, func_node):
        """Extract parameters from Tree-sitter node."""
        for child in func_node.children:
            if child.type == 'function_declarator':
                for grandchild in child.children:
                    if grandchild.type == 'parameter_list':
                        return grandchild.text.decode('utf-8')
        return ""
    
    def _extract_c_documentation_tree_sitter(self, file_content: str, function_line: int) -> Optional[str]:
        """Extract documentation comments preceding a function using Tree-sitter."""
        if not self.parsing_config.get('extract_comments', True):
            return None
        
        lines = file_content.splitlines()
        doc_lines = []
        line_idx = function_line - 2  # Start one line before function
        
        # Look backwards for documentation comments
        while line_idx >= 0:
            line = lines[line_idx].strip()
            if line.startswith('///') or line.startswith('/**') or line.startswith('*'):
                # Documentation comment
                clean_line = line.lstrip('/*').lstrip('*').rstrip('*/').strip()
                if clean_line:
                    doc_lines.insert(0, clean_line)
            elif line.startswith('//'):
                # Regular comment - might be documentation
                clean_line = line.lstrip('/').strip()
                if clean_line:
                    doc_lines.insert(0, clean_line)
            elif line == '':
                # Empty line - continue looking
                pass
            else:
                # Non-comment line - stop looking
                break
            line_idx -= 1
        
        return '\n'.join(doc_lines) if doc_lines else None
    
    def _parse_c_parameters(self, params: str) -> List[str]:
        """Parse C function parameters."""
        if not params.strip():
            return []
        
        arguments = []
        param_list = params.split(',')
        
        for param in param_list:
            param = param.strip()
            if param:
                # Extract just the parameter name (last word usually)
                parts = param.split()
                if parts:
                    # Handle cases like "int* ptr", "const char* str"
                    name = parts[-1].lstrip('*&')
                    arguments.append(name)
        
        return arguments
    
    def _is_function_modified_in_year(self, file_path: str, start_line: int, end_line: int, 
                                     year: int, git_cache: 'GitCache') -> bool:
        """Check if a function was introduced in the specified year (optimized for new functions only)."""
        relative_path = os.path.relpath(file_path, git_cache.repo_path)
        relative_path_normalized = relative_path.replace(os.sep, '/')
        
        # Get cached new functions for this year
        new_functions = git_cache.batch_get_new_functions_in_year(year, self.extensions)
        
        # Check if this file has any new functions
        if relative_path_normalized not in new_functions:
            return False
        
        # For now, if the file has new functions, we'll consider all functions in the file
        # as potentially new. This is a simplified approach for performance.
        # In a more sophisticated implementation, we could parse the function name
        # and check if it's in the new_functions set.
        return True


class CodeExtractor:
    """Extracts functions and methods from source files using language-specific parsers."""
    
    def __init__(self, repo_path: str = ".", max_workers: int = None, config: Dict[str, Any] = None):
        self.repo_path = Path(repo_path).resolve()
        self.original_repo_path = self.repo_path  # Store the original path
        self.functions = []
        self.original_commit = None
        self.temp_worktree = None
        self.skipped_functions_count = 0  # Statistics for skipped functions
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.git_cache = GitCache(str(self.original_repo_path))
        self.metrics = PerformanceMetrics()
        self._file_content_cache = {}  # Cache file contents
        
        # Initialize language configuration and parsers
        self.config = config or {}
        self._initialize_parsers()
        
    def _initialize_parsers(self):
        """Initialize language-specific parsers based on configuration."""
        self.parsers = {}
        enabled_languages = self.config.get('languages', {}).get('enabled', ['python'])
        
        # Create parsers for enabled languages
        if 'python' in enabled_languages:
            self.parsers['python'] = PythonParser(self.config)
        if 'c' in enabled_languages:
            self.parsers['c'] = CParser(self.config)
        
        # Build file extension to parser mapping
        self.extension_to_parser = {}
        for parser in self.parsers.values():
            for ext in parser.extensions:
                self.extension_to_parser[ext] = parser
        
        logger.info(f"Initialized parsers for languages: {list(self.parsers.keys())}")
        logger.info(f"Supported extensions: {list(self.extension_to_parser.keys())}")
    
    def get_parser_for_file(self, file_path: str) -> Optional[BaseParser]:
        """Get the appropriate parser for a file based on its extension."""
        for ext, parser in self.extension_to_parser.items():
            if file_path.endswith(ext):
                return parser
        return None
        
    def get_last_commit_of_year(self, year: int) -> Optional[str]:
        """Get the last commit hash of a specific year."""
        try:
            # Get the last commit of the specified year
            result = subprocess.run(
                ['git', 'log', '--format=%H', f'--since={year}-01-01', f'--until={year}-12-31', '-1'],
                cwd=self.original_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hash = result.stdout.strip()
            if commit_hash:
                logger.info(f"Found last commit of {year}: {commit_hash[:8]}")
                return commit_hash
            else:
                logger.warning(f"No commits found for year {year}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting last commit of {year}: {e}")
            return None
    
    def get_first_commit_of_year(self, year: int) -> Optional[str]:
        """Get the first commit hash of a specific year."""
        try:
            # Get the first commit of the specified year
            result = subprocess.run(
                ['git', 'log', '--reverse', '--format=%H', f'--since={year}-01-01', f'--until={year}-12-31'],
                cwd=self.original_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                commit_hash = lines[0].strip()
                logger.info(f"Found first commit of {year}: {commit_hash[:8]}")
                return commit_hash
            else:
                logger.warning(f"No commits found for year {year}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting first commit of {year}: {e}")
            return None
    
    def get_current_commit(self) -> Optional[str]:
        """Get the current commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.original_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting current commit: {e}")
            return None
    
    def create_temp_worktree(self, commit_hash: str) -> Optional[str]:
        """Create a temporary worktree for a specific commit."""
        try:
            # Create a temporary directory for the worktree
            temp_dir = tempfile.mkdtemp(prefix=f"git_worktree_{commit_hash[:8]}_")
            
            # Use the original repository path for git commands
            original_repo_path = self.original_repo_path
            
            logger.info(f"Creating worktree at {temp_dir}")
            
            # Create a new worktree
            result = subprocess.run(
                ['git', 'worktree', 'add', '--detach', temp_dir, commit_hash],
                cwd=original_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Created temporary worktree at {temp_dir} for commit {commit_hash[:8]}")
            return temp_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating worktree for commit {commit_hash[:8]}: {e}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
            return None
    
    def cleanup_temp_worktree(self):
        """Clean up the temporary worktree."""
        if self.temp_worktree:
            logger.info(f"Attempting to cleanup worktree: {self.temp_worktree}")
            
            try:
                # First, try to remove the worktree using git
                if os.path.exists(self.temp_worktree):
                    logger.info(f"Removing git worktree: {self.temp_worktree}")
                    subprocess.run(
                        ['git', 'worktree', 'remove', '--force', self.temp_worktree],
                        cwd=self.original_repo_path,
                        capture_output=True,
                        check=False
                    )
                    
                    # Wait a moment for the worktree to be fully removed
                    import time
                    time.sleep(2)
                    
                    # Remove the directory if it still exists
                    if os.path.exists(self.temp_worktree):
                        logger.info(f"Force removing directory: {self.temp_worktree}")
                        try:
                            # On Windows, we need to be more careful with directory removal
                            import shutil
                            shutil.rmtree(self.temp_worktree, ignore_errors=True)
                            
                            # Double-check if it's gone
                            if os.path.exists(self.temp_worktree):
                                logger.warning(f"Directory still exists after rmtree: {self.temp_worktree}")
                            else:
                                logger.info(f"Successfully removed worktree directory: {self.temp_worktree}")
                                
                        except Exception as e:
                            logger.error(f"Could not remove worktree directory {self.temp_worktree}: {e}")
                    else:
                        logger.info(f"Git worktree removal successful: {self.temp_worktree}")
                else:
                    logger.info(f"Worktree directory does not exist: {self.temp_worktree}")
                    
            except Exception as e:
                logger.error(f"Error during worktree cleanup: {e}")
            finally:
                self.temp_worktree = None
                # Reset repo_path back to original 
                self.repo_path = self.original_repo_path
                # Reset GitCache repo_path back to original
                self.git_cache.update_repo_path(str(self.original_repo_path))
                logger.info("Worktree cleanup completed")
    
    def cleanup_existing_worktrees(self):
        """Clean up any existing worktrees that might be left over."""
        try:
            logger.info("Checking for existing worktrees...")
            result = subprocess.run(
                ['git', 'worktree', 'list'],
                cwd=self.original_repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and '[detached' in line:
                        # Extract worktree path
                        parts = line.split()
                        if len(parts) >= 1:
                            worktree_path = parts[0]
                            if worktree_path != self.original_repo_path:
                                logger.info(f"Found existing worktree: {worktree_path}")
                                try:
                                    subprocess.run(
                                        ['git', 'worktree', 'remove', '--force', worktree_path],
                                        cwd=self.original_repo_path,
                                        capture_output=True,
                                        check=False
                                    )
                                    logger.info(f"Removed existing worktree: {worktree_path}")
                                except Exception as e:
                                    logger.warning(f"Could not remove existing worktree {worktree_path}: {e}")
            else:
                logger.info("No existing worktrees found")
                
        except Exception as e:
            logger.warning(f"Error checking existing worktrees: {e}")

    def setup_repository_for_year(self, year: int) -> bool:
        """Set up the repository to work with the specified year."""
        # Clean up any existing worktrees first
        self.cleanup_existing_worktrees()
        
        # Clean up any existing temporary worktree
        self.cleanup_temp_worktree()
        
        # Get the last commit of the specified year
        target_commit = self.get_last_commit_of_year(year)
        if not target_commit:
            raise Exception(f"No commits found for year {year}")
        
        # Store current commit
        self.original_commit = self.get_current_commit()
        
        # Try to create temporary worktree for the target commit
        self.temp_worktree = self.create_temp_worktree(target_commit)
        if not self.temp_worktree:
            raise Exception("Failed to create temporary worktree")
        
        # Update repo_path to point to the temporary worktree
        self.repo_path = Path(self.temp_worktree)
        self.git_cache.update_repo_path(str(self.repo_path)) # Update GitCache's repo_path
        logger.info(f"Repository set up for year {year} at commit {target_commit[:8]}")
        return True
    
    def get_first_repo_commit(self) -> Optional[int]:
        """Get the year of the first commit in the repository."""
        try:
            result = subprocess.run(
                ['git', '--no-pager', 'log', '--reverse', '--format=%cd', '--date=format:%Y'],
                cwd=self.original_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                year_str = lines[0].strip()
                return int(year_str)
            return None
                
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Error getting first commit year: {e}")
            return None
    
    def get_last_repo_commit(self) -> Optional[int]:
        """Get the year of the last commit in the repository."""
        try:
            result = subprocess.run(
                ['git', '--no-pager', 'log', '--format=%cd', '--date=format:%Y', '-1'],
                cwd=self.original_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            year_str = result.stdout.strip()
            if year_str:
                return int(year_str)
            return None
                
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Error getting last commit year: {e}")
            return None
    
    def get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        info = {
            'is_git_repo': True,  # Always true since we assume git repos only
            'current_commit': self.get_current_commit(),
            'first_commit_year': self.get_first_repo_commit(),
            'last_commit_year': self.get_last_repo_commit()
        }
        
        return info
    
    def get_file_commit_info(self, file_path: str, line_number: int) -> Dict[str, Any]:
        """Get commit information for a specific line in a file."""
        try:
            # Use cached git blame data
            relative_path = os.path.relpath(file_path, self.repo_path)
            commit_info = self.git_cache.get_line_commit_info(relative_path, line_number)
            
            if commit_info:
                self.metrics.record_cache_hit()
                return commit_info
            else:
                self.metrics.record_cache_miss()
                return {}
            
        except Exception as e:
            logger.warning(f"Error getting commit info for {file_path}:{line_number}: {e}")
            return {}
    
    def get_file_changes_in_year(self, file_path: str, year: int) -> List[Dict[str, Any]]:
        """Get all commits that modified a file in a specific year."""
        try:
            result = subprocess.run(
                ['git', 'log', '--format=%H|%an|%at|%s', f'--since={year}-01-01', f'--until={year}-12-31', '--', file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            changes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        timestamp = int(parts[2])
                        changes.append({
                            'commit_hash': parts[0],
                            'author': parts[1],
                            'date': datetime.fromtimestamp(timestamp).isoformat(),
                            'message': parts[3]
                        })
            
            return changes
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error getting changes for {file_path} in {year}: {e}")
            return []
    
    def is_function_modified_in_year(self, file_path: str, start_line: int, end_line: int, year: int) -> bool:
        """Check if a function (line range) was modified in the specified year."""
        # First check if file was modified at all in the year (cached operation)
        relative_path = os.path.relpath(file_path, self.repo_path)
        # Normalize path separators to forward slashes for git compatibility
        relative_path_normalized = relative_path.replace(os.sep, '/')
        
        if not self.git_cache.is_file_modified_in_year(relative_path_normalized, year, list(self.extension_to_parser.keys())):
            self.metrics.record_cache_hit()
            logger.debug(f"FILTERED: File {relative_path_normalized} was not modified in {year}")
            return False
        
        self.metrics.record_cache_miss()
        try:
            # Get commits that modified this line range in the specified year
            # Use original_repo_path to ensure we have complete git history
            # Use normalized path for git command
            result = subprocess.run(
                ['git', 'log', '--format=%H', f'--since={year}-01-01', f'--until={year}-12-31', 
                 '-L', f'{start_line},{end_line}:{relative_path_normalized}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.metrics.record_git_operation()
            was_modified = bool(result.stdout.strip())
            if not was_modified:
                logger.debug(f"FILTERED: Function in {relative_path_normalized}:{start_line}-{end_line} was not modified in {year}")
            return was_modified
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error checking if function was modified in {year}: {e}")
            return False
    

    

    

    
    def visit_file(self, file_path: str, target_year: Optional[int] = None) -> List[FunctionInfo]:
        """Parse a source file and extract all functions and methods using appropriate parser."""
        file_start_time = time.time()
        file_functions = []
        
        try:
            # Get the appropriate parser for this file
            parser = self.get_parser_for_file(file_path)
            if not parser:
                logger.debug(f"No parser available for file: {file_path}")
                return []
            
            # Read file content (with caching)
            if file_path not in self._file_content_cache:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self._file_content_cache[file_path] = content
            else:
                content = self._file_content_cache[file_path]
            
            # Use the appropriate parser to extract functions
            file_functions = parser.extract_functions(file_path, content, self.git_cache, target_year)
            
            # Record performance metrics
            processing_time = time.time() - file_start_time
            self.metrics.record_file_processed(processing_time)
            
            logger.debug(f"Processed {file_path} with {parser.language} parser: found {len(file_functions)} functions")
            return file_functions
                    
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return []
    
    def scan_repository(self, year: Optional[int] = None) -> List[FunctionInfo]:
        """Recursively scan the repository for Python files and extract functions."""
        logger.info(f"Scanning repository: {self.repo_path} with {self.max_workers} workers")
        
        try:
            # Find all supported source files efficiently
            source_files = []
            supported_extensions = list(self.extension_to_parser.keys())
            
            for root, dirs, files in os.walk(self.repo_path):
                # Skip common directories to ignore
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                
                for file in files:
                    if any(file.endswith(ext) for ext in supported_extensions):
                        source_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(source_files)} source files with extensions: {supported_extensions}")
            
            # If filtering by year, pre-cache modified files for the entire year
            if year:
                logger.info(f"Pre-caching modified files for year {year}...")
                self.git_cache.batch_get_modified_files_in_year(year, supported_extensions)
                try:
                    self.git_cache.batch_get_new_functions_in_year(year, supported_extensions)
                except Exception as e:
                    logger.error(f"Error getting new functions for year {year}: {e}")
            
            # Process files in parallel
            logger.info(f"Processing files with {self.max_workers} parallel workers...")
            all_functions = []
            
            if len(source_files) < 10:  # For small repos, don't use parallel processing
                for file_path in source_files:
                    logger.debug(f"Processing: {file_path}")
                    file_functions = self.visit_file(file_path, target_year=year)
                    all_functions.extend(file_functions)
            else:
                # Use ThreadPoolExecutor for I/O bound tasks like file reading and git operations
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all file processing tasks
                    future_to_file = {
                        executor.submit(self.visit_file, file_path, year): file_path 
                        for file_path in source_files
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            file_functions = future.result()
                            all_functions.extend(file_functions)
                            if len(all_functions) % 100 == 0:  # Progress indicator
                                logger.info(f"Processed {len([f for f in future_to_file if f.done()])} files, found {len(all_functions)} functions so far...")
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
            
            # Update main functions list
            self.functions = all_functions
            
            # Show performance metrics
            metrics = self.metrics.get_summary()
            logger.info(f"=== Performance Summary ===")
            logger.info(f"Extracted {len(self.functions)} functions/methods")
            logger.info(f"Files processed: {metrics['files_processed']}")
            logger.info(f"Processing time: {metrics['elapsed_time']:.2f}s")
            logger.info(f"Files per second: {metrics['files_per_second']:.2f}")
            logger.info(f"Functions per second: {metrics['functions_per_second']:.2f}")
            logger.info(f"Cache hit ratio: {metrics['cache_hit_ratio']:.2%}")
            logger.info(f"Git operations: {metrics['git_operations']}")
            
            if self.skipped_functions_count > 0:
                logger.info(f"Skipped {self.skipped_functions_count} functions that were not modified in target year")
            
            return self.functions
            
        finally:
            # Clean up temporary worktree if it was created
            self.cleanup_temp_worktree()

    def reset_state(self):
        """Reset the extractor state for processing a new year."""
        self.functions = []
        self.skipped_functions_count = 0
        # Clear caches to free memory
        self._file_content_cache.clear()
        self.git_cache.clear_cache()
        self.metrics = PerformanceMetrics()
        # Don't reset repo_path as it will be set by setup_repository_for_year
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage of caches."""
        return {
            'file_content_cache_entries': len(self._file_content_cache),
            'git_blame_cache_entries': len(self.git_cache.file_blame_cache),
            'git_modified_files_cache_entries': len(self.git_cache.modified_files_cache)
        }


class DuckDBManager:
    """Manages DuckDB operations for storing function information."""
    
    def __init__(self, db_path: str = "functions.db", repo_name: str = "default", year: int = datetime.now().year, drop_existing: bool = False):
        self.db_path = db_path
        self.repo_name = self._sanitize_table_name(repo_name)
        self.year = year
        
        # Optimize DuckDB connection for performance
        self.conn = duckdb.connect(db_path)
        self.conn.execute("PRAGMA threads=4")  # Use multiple threads
        self.conn.execute("PRAGMA memory_limit='2GB'")  # Set memory limit
        self.conn.execute("PRAGMA temp_directory='/tmp'")  # Use fast temp storage
        
        self.create_tables(drop_existing)
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name to ensure it's valid for SQL."""
        # Replace invalid characters with underscores
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in name.lower())
        # Remove leading/trailing underscores and multiple consecutive underscores
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = 'repo_' + sanitized
        # Limit length to avoid issues
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized or 'default'
    
    def _get_table_name(self, base_name: str) -> str:
        """Generate table name with repo and year prefix."""
        return f"{self.repo_name}_{self.year}_{base_name}"
    
    def _get_index_name(self, base_name: str) -> str:
        """Generate index name with repo and year prefix."""
        return f"idx_{self.repo_name}_{self.year}_{base_name}"
    
    def create_tables(self, drop_existing=False):
        """Create the necessary tables if they don't exist."""
        functions_table = self._get_table_name("functions")
        file_metadata_table = self._get_table_name("file_metadata")
        sequence_name = f"{self.repo_name}_{self.year}_functions_id_seq"
        
        if drop_existing:
            # Drop table and sequence if they exist to ensure schema is correct
            self.conn.execute(f"DROP TABLE IF EXISTS {functions_table}")
            self.conn.execute(f"DROP SEQUENCE IF EXISTS {sequence_name}")
        
        # Create sequence for auto-incrementing IDs
        self.conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {sequence_name}")
        
        # Main functions table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {functions_table} (
                id INTEGER PRIMARY KEY DEFAULT nextval('{sequence_name}'),
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
                language VARCHAR NOT NULL DEFAULT 'python',
                namespace VARCHAR,
                template_parameters TEXT,
                commit_hash VARCHAR,
                commit_date VARCHAR,
                commit_author VARCHAR,
                commit_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # File metadata table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {file_metadata_table} (
                file_path VARCHAR PRIMARY KEY,
                last_modified TIMESTAMP,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self._get_index_name('functions_name')} ON {functions_table}(name)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self._get_index_name('functions_file_path')} ON {functions_table}(file_path)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self._get_index_name('functions_type')} ON {functions_table}(function_type)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self._get_index_name('functions_language')} ON {functions_table}(language)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self._get_index_name('functions_commit_hash')} ON {functions_table}(commit_hash)")
        
        logger.info(f"Database tables created successfully for {self.repo_name}_{self.year}")
    
    def insert_functions(self, functions: List[FunctionInfo], batch_size: int = 1000):
        """Insert functions into the database with optimized batch processing."""
        if not functions:
            logger.warning("No functions to insert")
            return
        
        functions_table = self._get_table_name("functions")
        
        logger.info(f"Inserting {len(functions)} functions in batches of {batch_size}")
        
        # Begin transaction for better performance
        self.conn.begin()
        
        try:
            # Process in batches to manage memory
            for i in range(0, len(functions), batch_size):
                batch = functions[i:i + batch_size]
                
                # Prepare data for insertion
                data = []
                for func in batch:
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
                        func.language,
                        func.namespace,
                        func.template_parameters,
                        func.commit_hash,
                        func.commit_date,
                        func.commit_author,
                        func.commit_message
                    ))
                
                # Insert batch
                self.conn.executemany(f"""
                    INSERT INTO {functions_table} (
                        name, file_path, line_number, end_line, function_type, class_name,
                        docstring, signature, source_code, decorators, arguments, return_annotation,
                        is_async, is_generator, language, namespace, template_parameters,
                        commit_hash, commit_date, commit_author, commit_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, data)
                
                logger.debug(f"Inserted batch {i//batch_size + 1}/{(len(functions) + batch_size - 1)//batch_size}")
            
            # Commit transaction
            self.conn.commit()
            logger.info(f"Successfully inserted {len(functions)} functions into {functions_table}")
            
        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error inserting functions: {e}")
            raise
    
    def get_function_stats(self, year: int = None) -> Dict[str, Any]:
        """Get statistics about stored functions."""
        if year is None:
            year = self.year
        functions_table = self._get_table_name("functions")
        stats = {}
        
        # Total functions
        result = self.conn.execute(f"SELECT COUNT(*) FROM {functions_table}").fetchone()
        stats['total_functions'] = result[0] if result else 0
        
        # Functions by type
        result = self.conn.execute(f"""
            SELECT function_type, COUNT(*) 
            FROM {functions_table} 
            GROUP BY function_type
        """).fetchall()
        stats['by_type'] = dict(result)
        
        # Functions by file
        result = self.conn.execute(f"""
            SELECT file_path, COUNT(*) 
            FROM {functions_table} 
            GROUP BY file_path 
            ORDER BY COUNT(*) DESC 
            LIMIT 10
        """).fetchall()
        stats['top_files'] = dict(result)
        
        # Unique files
        result = self.conn.execute(f"SELECT COUNT(DISTINCT file_path) FROM {functions_table}").fetchone()
        stats['unique_files'] = result[0] if result else 0
        
        return stats
    
    def search_functions(self, query: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search functions by name or content."""
        functions_table = self._get_table_name("functions")
        
        where_conditions = []
        params = []
        
        if query:
            where_conditions.append("(name ILIKE ? OR docstring ILIKE ? OR source_code ILIKE ?)")
            params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        result = self.conn.execute(f"""
            SELECT name, file_path, line_number, function_type, class_name, docstring, 
                   commit_hash, commit_date, commit_author
            FROM {functions_table} 
            WHERE {where_clause}
            ORDER BY name
            LIMIT ?
        """, params + [limit]).fetchall()
        
        return [
            {
                'name': row[0],
                'file_path': row[1],
                'line_number': row[2],
                'function_type': row[3],
                'class_name': row[4],
                'docstring': row[5],
                'commit_hash': row[6],
                'commit_date': row[7],
                'commit_author': row[8]
            }
            for row in result
        ]
    
    def get_current_table_names(self) -> Dict[str, str]:
        """Get the current table names being used."""
        return {
            'functions': self._get_table_name("functions"),
            'file_metadata': self._get_table_name("file_metadata"),
            'sequence': f"{self.repo_name}_{self.year}_functions_id_seq"
        }
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get information about the current repository and year."""
        return {
            'repo_name': self.repo_name,
            'year': self.year,
            'table_prefix': f"{self.repo_name}_{self.year}",
            'database_path': self.db_path
        }
    
    def tables_exist(self) -> bool:
        """Check if tables exist for the current repo/year combination."""
        functions_table = self._get_table_name("functions")
        try:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {functions_table} LIMIT 1").fetchone()
            return True
        except:
            return False
    
    def list_all_tables(self) -> List[str]:
        """List all tables in the database."""
        result = self.conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result] if result else []
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


def handle_search_and_stats(db: DuckDBManager, search: str = None, stats: bool = False):
    """Handle search and statistics operations."""
    if stats:
        stats_data = db.get_function_stats(db.year)
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
            if result['commit_hash']:
                print(f"  Commit: {result['commit_hash'][:8]} by {result['commit_author']}")
            if result['commit_date']:
                print(f"  Date: {result['commit_date']}")
            if result['docstring']:
                print(f"  Docstring: {result['docstring'][:100]}...")


def visit_repo(extractor: CodeExtractor, year: int, repo_name: str, db_path: str, drop_existing: bool = False) -> Dict[str, Any]:
    """Process a single year of the repository and return statistics."""
    logger.info(f"Processing year {year}...")
    
    # Show git repository information for this year
    print(f"\n=== Processing Year {year} ===")
    
    # Show memory usage before processing
    memory_usage = extractor.get_memory_usage()
    logger.info(f"Cache entries before processing: {memory_usage}")
    
    # Extract functions for this specific year
    start_time = time.time()
    functions = extractor.scan_repository(year)
    processing_time = time.time() - start_time
    
    if not functions:
        logger.warning(f"No functions found for year {year}")
        return {
            'year': year,
            'total_functions': 0,
            'unique_files': 0,
            'by_type': {},
            'processing_time': processing_time,
            'success': False
        }
    
    # Store in DuckDB for this year
    db = DuckDBManager(db_path, repo_name=repo_name, year=year, drop_existing=drop_existing)
    
    # Show repository and table information for this year
    repo_info = db.get_repository_info()
    table_names = db.get_current_table_names()
    
    print(f"\n=== Repository Information for {year} ===")
    print(f"Repository: {repo_info['repo_name']}")
    print(f"Year: {repo_info['year']}")
    print(f"Database: {repo_info['database_path']}")
    print(f"Table prefix: {repo_info['table_prefix']}")
    
    print(f"\n=== Using Tables for {year} ===")
    print(f"Functions table: {table_names['functions']}")
    print(f"File metadata table: {table_names['file_metadata']}")
    print(f"Sequence: {table_names['sequence']}")
    
    # Insert with optimized batch processing
    insert_start = time.time()
    db.insert_functions(functions, batch_size=2000)  # Larger batches for better performance
    insert_time = time.time() - insert_start
    
    # Get statistics for this year
    stats_data = db.get_function_stats()
    print(f"\n=== Year {year} Complete ===")
    print(f"Total functions extracted: {stats_data['total_functions']}")
    print(f"Files processed: {stats_data['unique_files']}")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Database insert time: {insert_time:.2f}s")
    print(f"Functions by type:")
    for func_type, count in stats_data['by_type'].items():
        print(f"  {func_type}: {count}")
    
    # Show final memory usage
    final_memory_usage = extractor.get_memory_usage()
    logger.info(f"Cache entries after processing: {final_memory_usage}")
    
    db.close()
    
    return {
        'year': year,
        'total_functions': stats_data['total_functions'],
        'unique_files': stats_data['unique_files'],
        'by_type': stats_data['by_type'],
        'processing_time': processing_time,
        'insert_time': insert_time,
        'success': True
    }


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
    repo_name = config.get('repository', {}).get('name', 'default')
    year = config.get('repository', {}).get('year', datetime.now().year)
    start_year = config.get('repository', {}).get('start_year')
    end_year = config.get('repository', {}).get('end_year')
    search = config.get('repository', {}).get('search')
    stats = config.get('repository', {}).get('stats', False)
    
    if search or stats:
        # Just query the database
        db = DuckDBManager(db_path, repo_name=repo_name, year=year, drop_existing=False)
        handle_search_and_stats(db, search=search, stats=stats)
        db.close()
        return
    
    # Extract functions from repository
    logger.info("Starting optimized function extraction...")
    
    # Determine optimal worker count based on CPU cores
    cpu_count = os.cpu_count()
    optimal_workers = min(16, max(4, cpu_count))  # Use CPU cores, min 4, max 16
    logger.info(f"System has {cpu_count} CPU cores, using {optimal_workers} workers")
    
    extractor = CodeExtractor(repo_path, max_workers=optimal_workers, config=config)
    
    # Verify this is a git repository (one-time check)
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            raise Exception("Not a git repository")
    except (subprocess.SubprocessError, FileNotFoundError):
        raise Exception("Git not available or not a git repository")
    
    # Get all years with commits
    git_info = extractor.get_git_info()
    git_first_year = git_info.get('first_commit_year')
    git_last_year = git_info.get('last_commit_year')
    
    if not git_first_year or not git_last_year:
        raise Exception("Could not determine repository commit years")
    
    # Determine processing year range based on configuration
    if start_year is not None and end_year is not None:
        # Both start and end years are configured
        if start_year > end_year:
            raise Exception(f"Invalid year range: start_year ({start_year}) cannot be greater than end_year ({end_year})")
        first_year = max(start_year, git_first_year)
        last_year = min(end_year, git_last_year)
        year_source = "configured"
    elif start_year is not None:
        # Only start year is configured
        first_year = max(start_year, git_first_year)
        last_year = git_last_year
        year_source = "configured start year"
    elif end_year is not None:
        # Only end year is configured
        first_year = git_first_year
        last_year = min(end_year, git_last_year)
        year_source = "configured end year"
    else:
        # Use git-detected range (default behavior)
        first_year = git_first_year
        last_year = git_last_year
        year_source = "git-detected"
    
    print(f"\n=== Git Repository Information ===")
    print(f"Repository: {repo_name}")
    print(f"Path: {repo_path}")
    print(f"Git first commit year: {git_first_year}")
    print(f"Git last commit year: {git_last_year}")
    print(f"Processing year range: {first_year} to {last_year} ({year_source})")
    print(f"Years to process: {last_year - first_year + 1}")
    
    # Process each year
    all_stats = []
    successful_years = 0
    total_functions = 0
    
    for current_year in range(first_year, last_year + 1):
        try:
            extractor.reset_state()  # Reset state for new year
            extractor.setup_repository_for_year(current_year)
            year_stats = visit_repo(extractor, current_year, repo_name, db_path, drop_existing)
            all_stats.append(year_stats)
            
            if year_stats['success']:
                successful_years += 1
                total_functions += year_stats['total_functions']
            
        except Exception as e:
            logger.error(f"Error processing year {current_year}: {e}")
            all_stats.append({
                'year': current_year,
                'total_functions': 0,
                'unique_files': 0,
                'by_type': {},
                'success': False,
                'error': str(e)
            })
        finally:
            # Ensure cleanup after each year
            extractor.cleanup_temp_worktree()
    
    # Final cleanup to ensure no worktrees are left
    extractor.cleanup_temp_worktree()
    
    # Show overall statistics
    print(f"\n=== Overall Extraction Summary ===")
    print(f"Years processed: {len(all_stats)}")
    print(f"Successful years: {successful_years}")
    print(f"Total functions across all years: {total_functions}")
    
    print(f"\n=== Year-by-Year Summary ===")
    for year_stat in all_stats:
        status = "✓" if year_stat['success'] else "✗"
        error_msg = f" (Error: {year_stat.get('error', 'Unknown')})" if not year_stat['success'] else ""
        print(f"{status} {year_stat['year']}: {year_stat['total_functions']} functions{error_msg}")
    
    # Show aggregated statistics by function type
    print(f"\n=== Aggregated Functions by Type ===")
    type_totals = {}
    for year_stat in all_stats:
        if year_stat['success']:
            for func_type, count in year_stat['by_type'].items():
                type_totals[func_type] = type_totals.get(func_type, 0) + count
    
    for func_type, count in sorted(type_totals.items()):
        print(f"  {func_type}: {count}")
    
    logger.info("Function extraction completed successfully")


if __name__ == "__main__":
    main() 