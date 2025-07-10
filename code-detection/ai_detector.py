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
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import duckdb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Add the code-detection directory to the path to import baselines
sys.path.append('code-detection')

from baselines.rank import get_ranks, get_rank
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
    """Loads functions from DuckDB database."""
    
    def __init__(self, db_path: str = "functions.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
    
    def get_functions(self, limit: Optional[int] = None, 
                     function_type: Optional[str] = None,
                     file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load functions from database with optional filters."""
        
        query = "SELECT id, name, file_path, line_number, function_type, class_name, source_code FROM functions"
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
        
        # Unique files
        result = self.conn.execute("SELECT COUNT(DISTINCT file_path) FROM functions").fetchone()
        stats['unique_files'] = result[0] if result else 0
        
        return stats
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


class AIDetector:
    """Applies AI detection methods to functions."""
    
    def __init__(self, args):
        self.args = args
        self.model_config = {}
        
        # Setup models
        self._setup_models()
    
    def _setup_models(self):
        """Setup the base model and mask filling model."""
        logger.info("Setting up models...")
        
        # Preprocess and save
        cache_dir, base_model_name, SAVE_FOLDER = preprocess_and_save(self.args)
        self.model_config['cache_dir'] = cache_dir
        
        # Load mask filling model
        self.model_config = load_mask_filling_model(self.args, self.args.mask_filling_model_name, self.model_config)
        
        # Load base model
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
        
        # Vectorized masking
        if self.args.perturb_type == 'random':
            masked_texts = self._tokenize_and_mask_batch(texts, span_length, pct, ceil_pct)
        elif self.args.perturb_type == 'identifier-masking':
            masked_texts = self._tokenize_and_mask_identifiers_batch(texts, span_length, pct, ceil_pct)
        else:
            raise ValueError(f'Unknown perturb_type: {self.args.perturb_type}')
        
        # Batch model inference
        raw_fills = self._replace_masks_batch(masked_texts)
        extracted_fills = self._extract_fills_batch(raw_fills)
        perturbed_texts = self._apply_extracted_fills_batch(masked_texts, extracted_fills)
        
        return perturbed_texts
    
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
            batch_ranks = get_ranks(batch_codes, self.args, self.model_config, log=True)
            original_ranks.extend(batch_ranks)
        
        # Apply perturbations with optimized method
        logger.info("Applying perturbations...")
        perturbed_codes = self.perturb_texts_optimized([code for code in source_codes for _ in range(n_perturbations)])
        
        # Calculate perturbed log ranks in batches
        logger.info("Calculating perturbed log ranks...")
        perturbed_ranks = []
        
        for i in tqdm(range(0, len(perturbed_codes), n_perturbations), desc="Computing perturbed log ranks"):
            chunk = perturbed_codes[i:i + n_perturbations]
            chunk_ranks = get_ranks(chunk, self.args, self.model_config, log=True)
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization info."""
        stats = {
            'optimizations_applied': [
                'Batch processing for model inference',
                'Vectorized text operations',
                'Cached regex patterns',
                'Parallel identifier extraction',
                'GPU memory optimization with torch.no_grad()',
                'Larger batch sizes for better GPU utilization'
            ],
            'memory_optimizations': [
                'Reduced memory allocations',
                'Filtered empty masks before processing',
                'Efficient numpy operations'
            ],
            'speed_improvements': [
                'Parallel processing for CPU-bound tasks',
                'Batch rank calculations',
                'Optimized mask replacement'
            ]
        }
        return stats


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
            
            # Filtering arguments
            self.limit = config.get('filtering', {}).get('limit')
            self.function_type = config.get('filtering', {}).get('function_type')
            self.file_pattern = config.get('filtering', {}).get('file_pattern')
            
            # Model arguments
            self.base_model_name = config.get('models', {}).get('base_model_name', 'codellama/CodeLlama-7b-hf')
            self.mask_filling_model_name = config.get('models', {}).get('mask_filling_model_name', 'Salesforce/codet5p-770m')
            self.DEVICE = config.get('models', {}).get('device', 'cuda')
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
        
        print(f"  DetectCodeGPT Score: {result['detectcodegpt_score']:.4f}")
        print(f"  Original Rank: {result['original_rank']:.4f}")
        print(f"  Perturbed Rank Mean: {result['perturbed_rank_mean']:.4f}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    detectcodegpt_scores = [r['detectcodegpt_score'] for r in results]
    
    print(f"Number of functions processed: {len(results)}")
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
    
    logger.info("Starting AI detection process with OPTIMIZED perturbation...")
    
    # Load functions from database
    loader = FunctionLoader(args.db_path)
    
    # Show database statistics
    stats = loader.get_function_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total functions: {stats['total_functions']}")
    print(f"  Unique files: {stats['unique_files']}")
    print(f"  Functions by type: {stats['by_type']}")
    
    # Load functions with filters
    functions = loader.get_functions(
        limit=args.limit,
        function_type=args.function_type,
        file_pattern=args.file_pattern
    )
    
    if not functions:
        logger.warning("No functions found matching the criteria")
        loader.close()
        return
    
    print(f"\nProcessing {len(functions)} functions...")
    
    # Initialize AI detector
    detector = AIDetector(args)
    
    # Show optimization info
    perf_stats = detector.get_performance_stats()
    print(f"\n🚀 OPTIMIZATIONS APPLIED:")
    for opt in perf_stats['optimizations_applied']:
        print(f"  ✓ {opt}")
    
    print(f"\n⚡ SPEED IMPROVEMENTS:")
    for imp in perf_stats['speed_improvements']:
        print(f"  ✓ {imp}")
    
    print(f"\n💾 MEMORY OPTIMIZATIONS:")
    for mem in perf_stats['memory_optimizations']:
        print(f"  ✓ {mem}")
    
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