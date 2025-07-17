#!/usr/bin/env python3
"""
AI Detector - Function AI Detection Script

This script loads functions from the DuckDB database created by folder_lader.py
and applies AI detection methods to calculate detection scores.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import functools
import re
import scipy.stats
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import duckdb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Add the code-detection directory to the path to import baselines
sys.path.append('code-detection')

from baselines.rank_fast import get_ranks_fast, get_rank_fast
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.utils.preprocessing import preprocess_and_save
from identifier_tagging import get_identifier

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cache for compiled regex patterns
REGEX_CACHE = {
    'extra_id': re.compile(r"<extra_id_\d+>"),
    'extra_id_with_space': re.compile(r" <extra_id_\d+> "),
    'mask_string': re.compile(r"<<<mask>>>")
}


class FunctionLoader:
    """Loads functions from DuckDB database with year-specific tables."""
    
    def __init__(self, db_path: str = "functions.db", repo_name: str = "default", year: int = None):
        self.db_path = db_path
        self.repo_name = repo_name
        self.year = year
        self.conn = duckdb.connect(db_path)
        
        # If year is not specified, try to find available years
        if self.year is None:
            self.year = self._get_available_years()[0] if self._get_available_years() else None
    
    def _get_available_years(self) -> List[int]:
        """Get list of available years in the database."""
        try:
            # List all tables and extract years
            result = self.conn.execute("SHOW TABLES").fetchall()
            years = []
            for row in result:
                table_name = row[0]
                # Look for pattern: {repo_name}_{year}_functions
                if table_name.endswith('_functions'):
                    parts = table_name.split('_')
                    if len(parts) >= 2:
                        try:
                            year = int(parts[-2])  # Second to last part should be year
                            years.append(year)
                        except ValueError:
                            continue
            return sorted(years)
        except Exception as e:
            logger.error(f"Error getting available years: {e}")
            return []
    
    def _get_table_name(self, base_name: str) -> str:
        """Generate table name with repo and year prefix."""
        return f"{self.repo_name}_{self.year}_{base_name}"
    
    def get_functions(self, limit: Optional[int] = None, 
                     function_type: Optional[str] = None,
                     file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load functions from database with optional filters."""
        
        functions_table = self._get_table_name("functions")
        
        # Check if table exists
        try:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {functions_table} LIMIT 1").fetchone()
        except:
            logger.error(f"Table {functions_table} does not exist")
            return []
        
        query = f"SELECT id, name, file_path, line_number, function_type, class_name, source_code FROM {functions_table}"
        conditions = []
        params = []
        
        if function_type:
            conditions.append("function_type = ?")
            params.append(function_type)
        
        if file_pattern:
            conditions.append("file_path ILIKE ?")
            params.append(f"%{file_pattern}%")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY id"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = self.conn.execute(query, params).fetchall()
        
        functions = []
        for row in result:
            functions.append({
                'id': row[0],
                'name': row[1],
                'file_path': row[2],
                'line_number': row[3],
                'function_type': row[4],
                'class_name': row[5],
                'source_code': row[6]
            })
        
        return functions
    
    def get_function_stats(self) -> Dict[str, Any]:
        """Get statistics about stored functions."""
        functions_table = self._get_table_name("functions")
        stats = {}
        
        try:
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
            
            # Unique files
            result = self.conn.execute(f"SELECT COUNT(DISTINCT file_path) FROM {functions_table}").fetchone()
            stats['unique_files'] = result[0] if result else 0
            
            # Add repository and year info
            stats['repository'] = self.repo_name
            stats['year'] = self.year
            
        except Exception as e:
            logger.error(f"Error getting function stats: {e}")
            stats = {
                'total_functions': 0,
                'by_type': {},
                'unique_files': 0,
                'repository': self.repo_name,
                'year': self.year
            }
        
        return stats
    
    def list_available_repositories(self) -> List[Dict[str, Any]]:
        """List all available repositories and years in the database."""
        try:
            result = self.conn.execute("SHOW TABLES").fetchall()
            repos = {}
            
            for row in result:
                table_name = row[0]
                if table_name.endswith('_functions'):
                    parts = table_name.split('_')
                    if len(parts) >= 2:
                        try:
                            year = int(parts[-2])
                            repo_name = '_'.join(parts[:-2])  # Everything before year
                            if repo_name not in repos:
                                repos[repo_name] = []
                            repos[repo_name].append(year)
                        except ValueError:
                            continue
            
            # Convert to list format
            repo_list = []
            for repo_name, years in repos.items():
                repo_list.append({
                    'name': repo_name,
                    'years': sorted(years),
                    'total_years': len(years)
                })
            
            return repo_list
            
        except Exception as e:
            logger.error(f"Error listing repositories: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

    def get_functions_for_multiple_years(self, years: List[int], limit: Optional[int] = None, 
                                       function_type: Optional[str] = None,
                                       file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load functions from multiple years."""
        all_functions = []
        
        for year in years:
            self.year = year
            functions = self.get_functions(limit, function_type, file_pattern)
            # Add year information to each function
            for func in functions:
                func['year'] = year
            all_functions.extend(functions)
        
        return all_functions
    
    def get_function_stats_for_multiple_years(self, years: List[int]) -> Dict[str, Any]:
        """Get statistics for multiple years."""
        all_stats = []
        
        for year in years:
            self.year = year
            stats = self.get_function_stats()
            all_stats.append(stats)
        
        # Aggregate statistics
        total_functions = sum(stats['total_functions'] for stats in all_stats)
        total_files = sum(stats['unique_files'] for stats in all_stats)
        
        # Aggregate by type
        by_type = {}
        for stats in all_stats:
            for func_type, count in stats['by_type'].items():
                by_type[func_type] = by_type.get(func_type, 0) + count
        
        return {
            'total_functions': total_functions,
            'unique_files': total_files,
            'by_type': by_type,
            'repository': self.repo_name,
            'years': years,
            'year_stats': all_stats
        }


class AIDetector:
    """Applies AI detection methods to functions."""
    
    def __init__(self, args):
        self.args = args
        self.model_config = {}
        
        # Setup models
        self._setup_models()
    
    def _is_mask_model_required(self) -> bool:
        """Check if the current perturbation type requires the mask filling model."""
        mask_required_types = ['random', 'identifier-masking']
        return self.args.perturb_type in mask_required_types
    
    def _setup_models(self):
        """Setup the base model and mask filling model."""
        logger.info("Setting up models...")
        
        # Preprocess and save
        cache_dir, base_model_name, SAVE_FOLDER = preprocess_and_save(self.args)
        self.model_config['cache_dir'] = cache_dir
        
        # Check if mask filling model is needed
        if self._is_mask_model_required():
            logger.info(f"Loading mask filling model for perturbation type: {self.args.perturb_type}")
            # Load mask filling model
            self.model_config = load_mask_filling_model(self.args, self.args.mask_filling_model_name, self.model_config)
        else:
            logger.info(f"Mask filling model not required for perturbation type: {self.args.perturb_type}")
            logger.info("Skipping mask model loading to save memory and startup time")
        
        # Load base model (always required for rank calculation)
        logger.info("Loading base model for rank calculation...")
        self.model_config = load_base_model_and_tokenizer(self.args, self.model_config)
        
        logger.info("Models loaded successfully")
    
    def perturb_texts_optimized(self, texts: List[str], n_perturbations: int = 10) -> List[str]:
        """Optimized version of perturbation with batch processing and caching."""
        
        def perturb_texts_once_optimized(texts, ceil_pct=False):
            chunk_size = self.args.chunk_size
            if '11b' in self.args.mask_filling_model_name:
                chunk_size //= 2
            
            outputs = []
            # Process in larger batches for better GPU utilization
            batch_size = min(chunk_size * 2, len(texts))
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Applying perturbations (optimized)"):
                batch_texts = texts[i:i + batch_size]
                batch_outputs = self._perturb_texts_batch(batch_texts, ceil_pct)
                outputs.extend(batch_outputs)
            
            return outputs
        
        # Apply perturbations multiple times
        for i in range(self.args.n_perturbation_rounds):
            texts = perturb_texts_once_optimized(texts, ceil_pct=False)
        
        return texts
    
    def _perturb_texts_batch(self, texts: List[str], ceil_pct: bool = False) -> List[str]:
        """Optimized batch processing of perturbations."""
        span_length = self.args.span_length
        pct = self.args.pct_words_masked
        lambda_poisson = self.args.span_length
        
        # Vectorized masking
        if self.args.perturb_type == 'random':
            masked_texts = self._tokenize_and_mask_batch(texts, span_length, pct, ceil_pct)
        elif self.args.perturb_type == 'identifier-masking':
            masked_texts = self._tokenize_and_mask_identifiers_batch(texts, span_length, pct, ceil_pct)
        elif self.args.perturb_type == 'random-line-shuffle':
            perturbed_texts = [self._random_line_shuffle(x, pct) for x in texts]
            return perturbed_texts
        elif self.args.perturb_type == 'random-insert-newline':
            perturbed_texts = [self._random_insert_newline(x, pct, lambda_poisson) for x in texts]
            return perturbed_texts
        elif self.args.perturb_type == 'random-insert-space':
            perturbed_texts = [self._random_insert_space(x, pct, lambda_poisson) for x in texts]
            return perturbed_texts
        elif self.args.perturb_type == 'random-insert-space-newline':
            perturbed_texts = [self._random_insert_space(x, pct, lambda_poisson) for x in texts]
            perturbed_texts = [self._random_insert_newline(x, pct, lambda_poisson) for x in perturbed_texts]
            return perturbed_texts
        elif self.args.perturb_type == 'random-insert-space+newline':
            perturbed_texts_part1 = [self._random_insert_space(x, pct, lambda_poisson) for x in texts]
            perturbed_texts_part2 = [self._random_insert_newline(x, pct, lambda_poisson) for x in texts]
            total_num = len(perturbed_texts_part1)
            n1 = int(total_num / 2)
            n2 = total_num - n1
            perturbed_texts_part1 = perturbed_texts_part1[:n1]
            perturbed_texts_part2 = perturbed_texts_part2[:n2]
            return perturbed_texts_part1 + perturbed_texts_part2
        else:
            raise ValueError(f'Unknown perturb_type: {self.args.perturb_type}')
        
        # Batch model inference (only for masking-based perturbations)
        raw_fills = self._replace_masks_batch(masked_texts)
        extracted_fills = self._extract_fills_batch(raw_fills)
        perturbed_texts = self._apply_extracted_fills_batch(masked_texts, extracted_fills)
        
        return perturbed_texts
    
    def _random_line_shuffle(self, text: str, pct: float = 0.3) -> str:
        """Randomly exchange the order of two adjacent lines for pct of the lines, except for the first and last line."""
        lines = text.split('\n')
        n_lines = len(lines)
        n_shuffled = int(n_lines * pct)
        shuffled_idxs = np.random.choice(n_lines, n_shuffled, replace=False)
        for idx in shuffled_idxs:
            if idx == n_lines - 1 or idx == 0:
                continue
            lines[idx], lines[idx+1] = lines[idx+1], lines[idx]
        return '\n'.join(lines)
    
    def _random_insert_newline(self, text: str, pct: float = 0.3, mean: int = 1) -> str:
        """Randomly insert a newline for pct of the lines."""
        lines = text.split('\n')
        n_lines = len(lines)
        n_inserted = int(n_lines * pct)
        inserted_idxs = np.random.choice(n_lines, n_inserted, replace=False)
        for idx in inserted_idxs:
            n_newlines = 1
            lines[idx] = lines[idx] + '\n'*n_newlines
        return '\n'.join(lines)
    
    def _random_insert_space(self, text: str, pct: float = 0.3, mean: int = 1) -> str:
        """Randomly insert a space for pct of the lines."""
        tokens = text.split(' ')
        n_tokens = len(tokens)
        n_inserted = int(n_tokens * pct)
        inserted_idxs = np.random.choice(n_tokens, n_inserted, replace=False)
        for idx in inserted_idxs:
            n_spaces = scipy.stats.poisson.rvs(mean) + 1
            tokens[idx] = tokens[idx] + ' '*n_spaces
        return ' '.join(tokens)
    
    def _tokenize_and_mask_batch(self, texts: List[str], span_length: int, pct: float, ceil_pct: bool = False) -> List[str]:
        """Vectorized tokenization and masking."""
        results = []
        
        for text in texts:
            tokens = text.split(' ')
            mask_string = '<<<mask>>>'
            
            n_spans = pct * len(tokens) / (span_length + self.args.buffer_size * 2)
            if ceil_pct:
                n_spans = np.ceil(n_spans)
            n_spans = int(n_spans)
            
            # Use numpy for faster random operations
            if n_spans > 0:
                # Pre-calculate all possible positions
                valid_positions = []
                for start in range(len(tokens) - span_length):
                    end = start + span_length
                    search_start = max(0, start - self.args.buffer_size)
                    search_end = min(len(tokens), end + self.args.buffer_size)
                    if mask_string not in tokens[search_start:search_end]:
                        valid_positions.append(start)
                
                if valid_positions:
                    # Sample positions without replacement
                    selected_positions = np.random.choice(valid_positions, size=min(n_spans, len(valid_positions)), replace=False)
                    
                    # Sort in reverse order to avoid index shifting
                    selected_positions = np.sort(selected_positions)[::-1]
                    
                    for start in selected_positions:
                        tokens[start:start + span_length] = [mask_string]
            
            # Replace mask strings with extra_id tokens
            num_filled = 0
            for idx, token in enumerate(tokens):
                if token == mask_string:
                    tokens[idx] = f'<extra_id_{num_filled}>'
                    num_filled += 1
            
            results.append(' '.join(tokens))
        
        return results
    
    def _tokenize_and_mask_identifiers_batch(self, texts: List[str], span_length: int, pct: float, ceil_pct: bool = False) -> List[str]:
        """Optimized batch identifier masking with parallel processing."""
        results = []
        
        # Use ThreadPoolExecutor for CPU-bound identifier extraction
        with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
            # Extract identifiers in parallel
            identifier_futures = [executor.submit(get_identifier, text, 'python') for text in texts]
            identifier_results = [future.result() for future in identifier_futures]
        
        for text, (varnames, pos) in zip(texts, identifier_results):
            mask_string = ' <<<mask>>> '
            
            if not varnames:
                results.append(text)
                continue
            
            # Sample identifiers to mask
            n_to_mask = int(len(varnames) * self.args.pct_identifiers_masked)
            if n_to_mask > 0:
                sampled = np.random.choice(varnames, size=min(n_to_mask, len(varnames)), replace=False)
                sampled_set = set(sampled)
            else:
                sampled_set = set()
            
            # Split text into lines
            lines = text.split('\n')
            
            # Sort positions from end to start to avoid offset issues
            pos.sort(key=lambda pos: (-pos[0][0], -pos[0][1]))
            
            # Process each position
            for start, end in pos:
                line_number, start_pos = start
                _, end_pos = end
                
                identifier = lines[line_number][start_pos:end_pos]
                if identifier in sampled_set:
                    lines[line_number] = lines[line_number][:start_pos] + mask_string + lines[line_number][end_pos:]
            
            masked_text = '\n'.join(lines)
            tokens = masked_text.split(' ')
            
            # Replace mask strings with extra_id tokens
            num_filled = 0
            for idx, token in enumerate(tokens):
                if token == mask_string.strip():
                    tokens[idx] = f'<extra_id_{num_filled}>'
                    num_filled += 1
            
            text = ' '.join(tokens)
            
            # Remove spaces around masks using cached regex
            pattern_with_space = REGEX_CACHE['extra_id_with_space']
            matches = pattern_with_space.findall(text)
            for match in matches:
                text = text.replace(match, match.strip())
            
            results.append(text)
        
        return results
    
    def _replace_masks_batch(self, texts: List[str]) -> List[str]:
        """Optimized batch mask replacement with better GPU utilization."""
        pattern = REGEX_CACHE['extra_id']
        n_expected = [len(pattern.findall(x)) for x in texts]
        
        if max(n_expected) == 0:
            return texts
        
        # Safety check: ensure mask model is loaded
        if 'mask_model' not in self.model_config or 'mask_tokenizer' not in self.model_config:
            raise RuntimeError("Mask filling model not loaded but required for current perturbation type. "
                             "Please use 'random' or 'identifier-masking' perturbation types, or ensure mask model is loaded.")
        
        # Filter out texts with no masks to avoid unnecessary processing
        texts_with_masks = [(i, text) for i, text in enumerate(texts) if n_expected[i] > 0]
        
        if not texts_with_masks:
            return texts
        
        indices, masked_texts = zip(*texts_with_masks)
        
        # Truncate texts that are too long to prevent CUDA out of memory errors
        truncated_texts = []
        for text in masked_texts:
            # Simple truncation: split by words and limit to ~400 tokens to be safe
            words = text.split()
            if len(words) > 400:
                truncated_text = ' '.join(words[:400])
                logger.warning(f"Truncated text from {len(words)} to 400 words to prevent memory issues")
            else:
                truncated_text = text
            truncated_texts.append(truncated_text)
        
        stop_id = self.model_config['mask_tokenizer'].encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.model_config['mask_tokenizer'](truncated_texts, return_tensors="pt", padding=True).to(self.args.DEVICE)
        
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model_config['mask_model'].generate(
                **tokens, 
                max_length=512, 
                do_sample=True, 
                top_p=self.args.mask_top_p, 
                num_return_sequences=1, 
                eos_token_id=stop_id, 
                temperature=self.args.mask_temperature
            )
        
        generated_texts = self.model_config['mask_tokenizer'].batch_decode(outputs, skip_special_tokens=False)
        
        # Clean up GPU memory after processing
        del tokens, outputs
        torch.cuda.empty_cache()
        
        # Reconstruct full list
        result = texts.copy()
        for idx, generated_text in zip(indices, generated_texts):
            result[idx] = generated_text
        
        return result
    
    def _extract_fills_batch(self, texts: List[str]) -> List[List[str]]:
        """Optimized batch fill extraction."""
        pattern = REGEX_CACHE['extra_id']
        
        # Vectorized text cleaning
        cleaned_texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        
        # Extract text between mask tokens
        extracted_fills = [pattern.split(x)[1:-1] for x in cleaned_texts]
        
        # Remove whitespace around fills
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        
        return extracted_fills
    
    def _apply_extracted_fills_batch(self, masked_texts: List[str], extracted_fills: List[List[str]]) -> List[str]:
        """Optimized batch fill application."""
        pattern = REGEX_CACHE['extra_id']
        n_expected = [len(pattern.findall(x)) for x in masked_texts]
        
        texts = []
        for idx, (text, fills, n) in enumerate(zip(masked_texts, extracted_fills, n_expected)):
            if len(fills) < n:
                texts.append('')
            else:
                for fill_idx in range(n):
                    text = text.replace(f"<extra_id_{fill_idx}>", fills[fill_idx])
                texts.append(text)
        
        return texts
    
    # Keep original methods as fallbacks
    def perturb_texts(self, texts: List[str], n_perturbations: int = 10) -> List[str]:
        """Original perturbation method (kept for compatibility)."""
        return self.perturb_texts_optimized(texts, n_perturbations)
    
    def _perturb_texts_chunk(self, texts: List[str], ceil_pct: bool = False) -> List[str]:
        """Original chunk processing method (kept for compatibility)."""
        return self._perturb_texts_batch(texts, ceil_pct)
    
    def _tokenize_and_mask(self, text: str, span_length: int, pct: float, ceil_pct: bool = False) -> str:
        """Original single text masking method (kept for compatibility)."""
        return self._tokenize_and_mask_batch([text], span_length, pct, ceil_pct)[0]
    
    def _tokenize_and_mask_identifiers(self, text: str, span_length: int, pct: float, ceil_pct: bool = False) -> str:
        """Original single text identifier masking method (kept for compatibility)."""
        return self._tokenize_and_mask_identifiers_batch([text], span_length, pct, ceil_pct)[0]
    
    def _replace_masks(self, texts: List[str]) -> List[str]:
        """Original mask replacement method (kept for compatibility)."""
        return self._replace_masks_batch(texts)
    
    def _extract_fills(self, texts: List[str]) -> List[List[str]]:
        """Original fill extraction method (kept for compatibility)."""
        return self._extract_fills_batch(texts)
    
    def _apply_extracted_fills(self, masked_texts: List[str], extracted_fills: List[List[str]]) -> List[str]:
        """Original fill application method (kept for compatibility)."""
        return self._apply_extracted_fills_batch(masked_texts, extracted_fills)
    
    def calculate_scores_optimized(self, functions: List[Dict[str, Any]], n_perturbations: int = 10) -> List[Dict[str, Any]]:
        """Optimized version of calculate_scores with batch processing."""
        results = []
        
        # Extract source codes
        source_codes = [func['source_code'] for func in functions]
        
        logger.info(f"Processing {len(functions)} functions...")
        
        # Calculate unperturbed log ranks in batches
        logger.info("Calculating unperturbed log ranks...")
        original_ranks = []
        batch_size = self.args.batch_size
        
        for i in tqdm(range(0, len(source_codes), batch_size), desc="Computing unperturbed log ranks"):
            batch_codes = source_codes[i:i + batch_size]
            batch_ranks = get_ranks_fast(batch_codes, self.args, self.model_config, log=True, batch_size=batch_size)
            original_ranks.extend(batch_ranks)
        
        # Apply perturbations with optimized method
        logger.info("Applying perturbations...")
        perturbed_codes = self.perturb_texts_optimized([code for code in source_codes for _ in range(n_perturbations)])
        
        # Calculate perturbed log ranks in batches
        logger.info("Calculating perturbed log ranks...")
        perturbed_ranks = []
        
        for i in tqdm(range(0, len(perturbed_codes), n_perturbations), desc="Computing perturbed log ranks"):
            chunk = perturbed_codes[i:i + n_perturbations]
            chunk_ranks = get_ranks_fast(chunk, self.args, self.model_config, log=True, batch_size=len(chunk))
            perturbed_ranks.append(chunk_ranks)
        
        # Compile results
        for i, func in enumerate(functions):
            original_rank = original_ranks[i]
            p_ranks = perturbed_ranks[i]
            
            # Calculate perturbed rank mean
            p_rank_mean = np.mean([rank for rank in p_ranks if not math.isnan(rank)])
            
            # DetectCodeGPT score
            detectcodegpt_score = p_rank_mean / original_rank if original_rank != 0 else 0
            
            result = {
                'function_id': func['id'],
                'name': func['name'],
                'file_path': func['file_path'],
                'line_number': func['line_number'],
                'function_type': func['function_type'],
                'class_name': func['class_name'],
                'original_rank': original_rank,
                'perturbed_rank_mean': p_rank_mean,
                'detectcodegpt_score': detectcodegpt_score
            }
            
            results.append(result)
        
        return results

    def calculate_scores(self, functions: List[Dict[str, Any]], n_perturbations: int = 10) -> List[Dict[str, Any]]:
        """Calculate detection scores for functions (now uses optimized version)."""
        return self.calculate_scores_optimized(functions, n_perturbations)
    

def setup_args():
    """Setup and parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Detection for Functions")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to configuration file")
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return None


def create_args_from_config(config):
    """Create args object from configuration."""
    class Args:
        def __init__(self, config):
            # Database arguments
            self.db_path = config.get('database', {}).get('path', 'functions.db')
            self.repo_name = config.get('database', {}).get('repo_name', 'default')
            self.year = config.get('database', {}).get('year', None)
            
            # Filtering arguments
            self.limit = config.get('filtering', {}).get('limit', None)
            self.function_type = config.get('filtering', {}).get('function_type')
            self.file_pattern = config.get('filtering', {}).get('file_pattern')
            
            # Model arguments
            self.base_model_name = config.get('models', {}).get('base_model_name', 'codellama/CodeLlama-7b-hf')
            self.mask_filling_model_name = config.get('models', {}).get('mask_filling_model_name', 'Salesforce/codet5p-770m')
            
            # Handle device selection
            device_config = config.get('models', {}).get('device', 'auto')
            if device_config == 'auto':
                self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.DEVICE = device_config
            
            self.cache_dir = config.get('models', {}).get('cache_dir', '~/.cache/huggingface/hub')
            self.int8 = config.get('models', {}).get('int8', False)
            self.half = config.get('models', {}).get('half', False)
            self.base_half = config.get('models', {}).get('base_half', False)
            
            # Perturbation arguments
            self.pct_words_masked = config.get('perturbation', {}).get('pct_words_masked', 0.5)
            self.pct_identifiers_masked = config.get('perturbation', {}).get('pct_identifiers_masked', 0.75)
            self.span_length = config.get('perturbation', {}).get('span_length', 2)
            self.n_perturbation_rounds = config.get('perturbation', {}).get('n_perturbation_rounds', 1)
            self.perturb_type = config.get('perturbation', {}).get('perturb_type', 'random')
            self.buffer_size = config.get('perturbation', {}).get('buffer_size', 1)
            self.mask_top_p = config.get('perturbation', {}).get('mask_top_p', 1.0)
            self.mask_temperature = config.get('perturbation', {}).get('mask_temperature', 1.0)
            self.chunk_size = config.get('perturbation', {}).get('chunk_size', 10)
            self.n_perturbations = config.get('perturbation', {}).get('n_perturbations', 10)
            
            # Processing arguments
            self.batch_size = config.get('processing', {}).get('batch_size', 50)
            self.n_similarity_samples = config.get('processing', {}).get('n_similarity_samples', 20)
            
            # Output arguments
            self.output_name = config.get('output', {}).get('output_name', 'test_ipynb')
            self.visualize = config.get('output', {}).get('visualize', False)
            
            # Additional required arguments for preprocessing
            self.do_top_k = config.get('sampling', {}).get('do_top_k', False)
            self.do_top_p = config.get('sampling', {}).get('do_top_p', False)
            self.scoring_model_name = config.get('models', {}).get('scoring_model_name', '')
            self.n_samples = config.get('processing', {}).get('n_samples', 5)
            self.temperature = config.get('sampling', {}).get('temperature', 1.0)
            self.dataset = config.get('dataset', {}).get('name', 'functions')
            self.dataset_key = config.get('dataset', {}).get('key', '')
            self.min_len = config.get('processing', {}).get('min_len', 0)
            self.max_len = config.get('processing', {}).get('max_len', 128)
            
    
    return Args(config)


def print_results(results: List[Dict[str, Any]]):
    """Print detection results to screen."""
    print("\n" + "="*80)
    print("DETECTCODEGPT RESULTS")
    print("="*80)
    
    # Print individual function results
    for result in results:
        print(f"\nFunction: {result['name']}")
        print(f"  File: {result['file_path']}:{result['line_number']}")
        print(f"  Type: {result['function_type']}")
        if result['class_name']:
            print(f"  Class: {result['class_name']}")
        if 'year' in result:
            print(f"  Year: {result['year']}")
        
        print(f"  DetectCodeGPT Score: {result['detectcodegpt_score']:.4f}")
        print(f"  Original Rank: {result['original_rank']:.4f}")
        print(f"  Perturbed Rank Mean: {result['perturbed_rank_mean']:.4f}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    detectcodegpt_scores = [r['detectcodegpt_score'] for r in results]
    
    print(f"Number of functions processed: {len(results)}")
    
    # Show year breakdown if available
    if 'year' in results[0]:
        years = list(set(r['year'] for r in results))
        print(f"Years processed: {sorted(years)}")
        for year in sorted(years):
            year_results = [r for r in results if r['year'] == year]
            year_scores = [r['detectcodegpt_score'] for r in year_results]
            print(f"  {year}: {len(year_results)} functions, mean score: {np.mean(year_scores):.4f}")
    
    print(f"\nDetectCodeGPT Scores:")
    print(f"  Mean: {np.mean(detectcodegpt_scores):.4f}")
    print(f"  Std: {np.std(detectcodegpt_scores):.4f}")
    print(f"  Min: {np.min(detectcodegpt_scores):.4f}")
    print(f"  Max: {np.max(detectcodegpt_scores):.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  Higher scores indicate higher likelihood of being AI-generated")
    print(f"  Lower scores indicate higher likelihood of being human-written")
    print(f"  Score = Perturbed_Rank_Mean / Original_Rank")


def main():
    """Main function."""
    args = setup_args()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return
    
    # Create args object from config
    args = create_args_from_config(config)
    
    # Log device information
    if args.DEVICE == 'cuda':
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        logger.info(f"Starting AI detection process with CUDA: {device_name}")
    else:
        logger.info(f"Starting AI detection process with {args.DEVICE.upper()}")
        if args.DEVICE == 'cpu':
            logger.warning("CPU mode detected - processing will be significantly slower")
            logger.info("Consider reducing batch_size and n_perturbations for better performance")
    
    # Load functions from database
    loader = FunctionLoader(args.db_path, args.repo_name, args.year)
    
    # Show available repositories
    available_repos = loader.list_available_repositories()
    print(f"\nAvailable repositories in database:")
    for repo in available_repos:
        print(f"  {repo['name']}: {repo['years']} ({repo['total_years']} years)")
    
    # Determine which years to process
    if args.year is None:
        # Process all available years for the repository
        target_repo = next((repo for repo in available_repos if repo['name'] == args.repo_name), None)
        if target_repo:
            years_to_process = target_repo['years']
            print(f"\nProcessing all available years for {args.repo_name}: {years_to_process}")
        else:
            logger.error(f"Repository {args.repo_name} not found in database")
            return
    else:
        years_to_process = [args.year]
        print(f"\nProcessing specific year: {args.year}")
    
    # Show database statistics
    if len(years_to_process) == 1:
        stats = loader.get_function_stats()
        print(f"\nDatabase Statistics for {stats['repository']} ({stats['year']}):")
        print(f"  Total functions: {stats['total_functions']}")
        print(f"  Unique files: {stats['unique_files']}")
        print(f"  Functions by type: {stats['by_type']}")
    else:
        stats = loader.get_function_stats_for_multiple_years(years_to_process)
        print(f"\nDatabase Statistics for {stats['repository']} ({stats['years']}):")
        print(f"  Total functions: {stats['total_functions']}")
        print(f"  Unique files: {stats['unique_files']}")
        print(f"  Functions by type: {stats['by_type']}")
        print(f"  Year breakdown:")
        for year_stat in stats['year_stats']:
            print(f"    {year_stat['year']}: {year_stat['total_functions']} functions")
    
    # Load functions with filters
    if len(years_to_process) == 1:
        functions = loader.get_functions(
            limit=args.limit,
            function_type=args.function_type,
            file_pattern=args.file_pattern
        )
    else:
        functions = loader.get_functions_for_multiple_years(
            years_to_process,
            limit=args.limit,
            function_type=args.function_type,
            file_pattern=args.file_pattern
        )
    
    if not functions:
        raise ValueError("No functions found matching the criteria")
    
    print(f"\nProcessing {len(functions)} functions...")
    
    # Show model loading information
    mask_required_types = ['random', 'identifier-masking']
    if args.perturb_type in mask_required_types:
        print(f"Perturbation type '{args.perturb_type}' requires mask filling model")
        print(f"Will load: Base model + Mask filling model")
    else:
        print(f"Perturbation type '{args.perturb_type}' does not require mask filling model")
        print(f"Will load: Base model only (saving memory)")
    
    # Initialize AI detector
    detector = AIDetector(args)
    
    # Calculate scores with timing
    import time
    start_time = time.time()
    
    results = detector.calculate_scores(functions, args.n_perturbations)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    print_results(results)
    
    # Show performance metrics
    print(f"\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per function: {processing_time/len(functions):.3f} seconds")
    print(f"Functions processed per second: {len(functions)/processing_time:.2f}")
    
    if args.n_perturbations > 0:
        total_perturbations = len(functions) * args.n_perturbations
        print(f"Total perturbations: {total_perturbations}")
        print(f"Perturbations per second: {total_perturbations/processing_time:.2f}")
    
    # Cleanup
    loader.close()
    torch.cuda.empty_cache()
    
    logger.info("AI detection completed successfully with optimized performance!")


if __name__ == "__main__":
    main() 