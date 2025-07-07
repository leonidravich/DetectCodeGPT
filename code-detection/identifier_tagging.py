from transformers import RobertaTokenizer
import argparse
from loguru import logger
import os
from tqdm import tqdm
import json
import numpy as np
import pdb
import re

# Try to import tree-sitter, with fallback if not available
try:
    from tree_sitter import Language, Parser
    
    # Check if we have the tree-sitter grammars
    grammar_dirs = [
        './tree-sitter/tree-sitter-python',
        './tree-sitter/tree-sitter-java',
        './tree-sitter/tree-sitter-php',
        './tree-sitter/tree-sitter-go',
        './tree-sitter/tree-sitter-ruby',
        './tree-sitter/tree-sitter-javascript',
    ]
    
    # Check if any grammar directories exist
    grammar_exists = any(os.path.exists(grammar_dir) for grammar_dir in grammar_dirs)
    
    if grammar_exists and not os.path.exists('build/my-languages.so'):
        try:
            # Use the new tree-sitter API
            Language.build_library(
                # Store the library in the `build` directory
                'build/my-languages.so',
                # Include one or more languages
                [grammar_dir for grammar_dir in grammar_dirs if os.path.exists(grammar_dir)]
            )
        except Exception as e:
            logger.warning(f"Failed to build tree-sitter library: {e}")
            grammar_exists = False
    elif os.path.exists('build/my-languages.so'):
        logger.info('build/my-languages.so already exists, skip building')
    else:
        logger.warning("Tree-sitter grammar directories not found, using fallback method")
        grammar_exists = False
    
    if grammar_exists:
        PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
        JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
        PHP_LANGUAGE = Language('build/my-languages.so', 'php')
        GO_LANGUAGE = Language('build/my-languages.so', 'go')
        RUBY_LANGUAGE = Language('build/my-languages.so', 'ruby')
        JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')

        # map from language to tree-sitter language
        LANGUAGE_MAP = {
            'java': JAVA_LANGUAGE,
            'python': PYTHON_LANGUAGE,
            'php': PHP_LANGUAGE,
            'go': GO_LANGUAGE,
            'ruby': RUBY_LANGUAGE,
            'javascript': JAVASCRIPT_LANGUAGE,
        }
        parser = Parser()
        TREE_SITTER_AVAILABLE = True
    else:
        TREE_SITTER_AVAILABLE = False
        
except ImportError:
    logger.warning("tree-sitter not available, using fallback method")
    TREE_SITTER_AVAILABLE = False


def get_identifier_fallback(code, lang):
    """Fallback method using regex to extract identifiers when tree-sitter is not available."""
    identifiers = []
    pos = []
    
    # Python identifier pattern
    if lang == 'python':
        # Pattern for Python identifiers (variable names, function names, etc.)
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = list(re.finditer(pattern, code))
        
        for match in matches:
            identifier = match.group()
            # Skip Python keywords and common built-ins
            python_keywords = {
                'False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 
                'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 
                'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 
                'raise', 'return', 'try', 'while', 'with', 'yield', 'self', 'cls'
            }
            
            if identifier not in python_keywords and not identifier.startswith('__'):
                # Calculate position
                start_line = code[:match.start()].count('\n')
                start_col = match.start() - code.rfind('\n', 0, match.start()) - 1
                end_line = code[:match.end()].count('\n')
                end_col = match.end() - code.rfind('\n', 0, match.end()) - 1
                
                pos.append(((start_line, start_col), (end_line, end_col)))
                identifiers.append(identifier)
    
    # Java identifier pattern
    elif lang == 'java':
        pattern = r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b'
        matches = list(re.finditer(pattern, code))
        
        for match in matches:
            identifier = match.group()
            # Skip Java keywords
            java_keywords = {
                'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 
                'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 
                'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 
                'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 
                'package', 'private', 'protected', 'public', 'return', 'short', 'static', 
                'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 
                'transient', 'try', 'void', 'volatile', 'while'
            }
            
            if identifier not in java_keywords:
                start_line = code[:match.start()].count('\n')
                start_col = match.start() - code.rfind('\n', 0, match.start()) - 1
                end_line = code[:match.end()].count('\n')
                end_col = match.end() - code.rfind('\n', 0, match.end()) - 1
                
                pos.append(((start_line, start_col), (end_line, end_col)))
                identifiers.append(identifier)
    
    # Generic fallback for other languages
    else:
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = list(re.finditer(pattern, code))
        
        for match in matches:
            identifier = match.group()
            start_line = code[:match.start()].count('\n')
            start_col = match.start() - code.rfind('\n', 0, match.start()) - 1
            end_line = code[:match.end()].count('\n')
            end_col = match.end() - code.rfind('\n', 0, match.end()) - 1
            
            pos.append(((start_line, start_col), (end_line, end_col)))
            identifiers.append(identifier)
    
    return list(set(identifiers)), pos


def get_identifier(code, lang):
    """Extract identifiers from code using tree-sitter or fallback method."""
    if TREE_SITTER_AVAILABLE and lang in LANGUAGE_MAP:
        return get_identifier_tree_sitter(code, lang)
    else:
        logger.info(f"Using fallback method for language: {lang}")
        return get_identifier_fallback(code, lang)


def get_identifier_tree_sitter(code, lang):
    """Original tree-sitter based identifier extraction."""
    pos = []
    identifiers = []

    def traverse(root):
        if root is None:
            return
        for child in root.children:
            if child.type == 'identifier':
                if child.start_byte > 0 and code[child.start_byte - 1] == '.':
                    continue
                # the token 'self' is not an identifier to perturb
                if get_identifier_from_position(code, child.start_point, child.end_point) == 'self':
                    continue
                
                start_point = child.start_point
                end_point = child.end_point
                pos.insert(0, (start_point, end_point))
            traverse(child)

    parser.set_language(LANGUAGE_MAP[lang])
    tree = parser.parse(bytes(code, 'utf-8'))
    traverse(tree.root_node)
    for id in pos:
        identifiers.append(get_identifier_from_position(code, id[0], id[1]))
    # return identifiers, and the position of all identifiers
    return list(set(identifiers)), pos


def get_identifier_from_position(code_string, start_point, end_point):
    lines = code_string.splitlines()
    identifier = lines[start_point[0]][start_point[1]:end_point[1]]
    return identifier


def load_data(path, language='python', max_num=10000):

    all_data = []
    path_to_data = f'{path}/{language}/train.jsonl'

    logger.info(f'Loading data from {path_to_data}')

    failed = 0
    success = 0

    max_len = 128
    min_len = 5

    with open(path_to_data, 'r') as f:
        count = 0
        for line in tqdm(f):

            if count >= max_num:
                break

            data = json.loads(line)
            data['original_string'] = data['original_string'].replace("'''", '"""')
            try:
                prompt = data['original_string'].split('"""')[0] + '"""' + data['original_string'].split('"""')[1] + '"""'
                solution = data['original_string'].split('"""')[2]
                success += 1
            except:
                failed += 1

            all_data.append(solution)
            count += 1

    logger.info(f'Failed: {failed}, Success: {success}')

    all_lengths = [len(data.split()) for data in all_data]
    logger.info(f'All lengths: min: {min(all_lengths)}, max: {max(all_lengths)}, mean: {np.mean(all_lengths)}, std: {np.std(all_lengths)}')

    return all_data

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--language_name', type=str, default='java')
    params = argparser.parse_args()

    # code_item = "def test():\n    print('hello world')"
    texts = ['''
def remove_mask_space(text, args, **kwargs):
    # find all the mask positions " <extra_id_\d+> ", and remove the space before and after the mask
    pattern = re.compile(r" <extra_id_\d+> ")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text
''']
    texts = ['''
tags to be added to the image.
        :type extra_tags: list of unicode | str
        :param add_latest: If True, the latest tag will be added to the list of tags.
        :type add_latest: bool
        :return: The image id.
        :rtype: unicode | str
        """
        if not isinstance(main_tag, six.string_types):
            raise TypeError('main_tag must be a string')
        if not isinstance(extra_tags, list):
''']
    code_item = texts[0]

    # max_num = 100
    # original_codes = load_data(path=path, language='python', max_num=max_num)
    # code_item = original_codes[0]

    varnames, pos = get_identifier(code_item, 'python')

    logger.info(f'code_item: \n{code_item}')
    logger.info(f'varnames: \n{varnames}')
    logger.info(f'pos: \n{pos}')
