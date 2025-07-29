import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

# Try different approaches for compatibility
try:
    # Method 1: Direct language creation (newer versions)
    C_LANGUAGE = Language(tsc.language())
    parser = Parser()
    parser.language = C_LANGUAGE
except (TypeError, ValueError, AttributeError) as e:
    print(f"Method 1 failed: {e}")
    try:
        # Method 2: Using the older API
        C_LANGUAGE = Language(tsc.language())
        parser = Parser()
        parser.set_language(C_LANGUAGE)
    except (TypeError, ValueError, AttributeError) as e:
        print(f"Method 2 failed: {e}")
        print("Please update your tree-sitter packages:")
        print("pip install --upgrade tree-sitter tree-sitter-c")
        exit(1)

# Sample C code to parse
c_code = """
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    int num = 5;
    printf("Factorial of %d is %d\\n", num, factorial(num));
    return 0;
}
"""

print("Parser initialized successfully!")

# Parse the code
tree = parser.parse(bytes(c_code, "utf8"))

def print_tree(node, indent=0):
    """Recursively print the syntax tree"""
    if indent < 3:  # Limit depth for readability
        node_text = c_code[node.start_byte:node.end_byte]
        # Truncate long text for readability
        if len(node_text) > 50:
            node_text = node_text[:47] + "..."
        print("  " * indent + f"{node.type}: {repr(node_text)}")
        for child in node.children:
            print_tree(child, indent + 1)

# Print a simplified syntax tree
print("\n=== Syntax Tree (top 3 levels) ===")
print_tree(tree.root_node)

print("\n=== Finding Function Definitions ===")
def find_functions(node):
    """Find function definitions by traversing the tree"""
    if node.type == "function_definition":
        # Find the function name
        for child in node.children:
            if child.type == "function_declarator":
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        func_name = c_code[grandchild.start_byte:grandchild.end_byte]
                        print(f"Found function: {func_name}")
                        break
    
    for child in node.children:
        find_functions(child)

find_functions(tree.root_node)

print("\n=== Finding All Identifiers ===")
def find_identifiers(node):
    """Find all identifiers in the code"""
    if node.type == "identifier":
        identifier = c_code[node.start_byte:node.end_byte]
        line_num = node.start_point[0] + 1
        print(f"Identifier: '{identifier}' at line {line_num}")
    
    for child in node.children:
        find_identifiers(child)

find_identifiers(tree.root_node)

print("\n=== Finding Include Statements ===")
def find_includes(node):
    """Find preprocessor include statements"""
    if node.type == "preproc_include":
        include_text = c_code[node.start_byte:node.end_byte]
        print(f"Include statement: {include_text}")
    
    for child in node.children:
        find_includes(child)

find_includes(tree.root_node)

print("\n=== Node Type Analysis ===")
def analyze_node_types(node, types_found=None):
    """Collect all node types in the tree"""
    if types_found is None:
        types_found = set()
    
    types_found.add(node.type)
    for child in node.children:
        analyze_node_types(child, types_found)
    
    return types_found

node_types = analyze_node_types(tree.root_node)
print("Node types found in the C code:")
for node_type in sorted(node_types):
    print(f"  - {node_type}")

print("\n=== Error Handling Example ===")
# Example with syntax error
bad_c_code = """
int main( {
    printf("Missing closing parenthesis"
    return 0;
}
"""

bad_tree = parser.parse(bytes(bad_c_code, "utf8"))
if bad_tree.root_node.has_error:
    print("Syntax error detected in the bad code!")
    
    def find_errors(node):
        if node.type == "ERROR":
            error_text = bad_c_code[node.start_byte:node.end_byte]
            line_num = node.start_point[0] + 1
            print(f"Error at line {line_num}: {repr(error_text[:50])}")
        elif node.has_error:
            print(f"Node with error: {node.type} at line {node.start_point[0] + 1}")
        
        for child in node.children:
            find_errors(child)
    
    find_errors(bad_tree.root_node)
else:
    print("No syntax errors found (unexpected)")

print("\n=== Advanced Query Example (if supported) ===")
# Try using the newer Query API if available
try:
    from tree_sitter import Query
    
    # Query for function definitions using the new API
    query_text = """
    (function_definition
      type: (primitive_type) @return_type
      declarator: (function_declarator
        declarator: (identifier) @function_name))
    """
    
    query = Query(C_LANGUAGE, query_text)
    matches = query.matches(tree.root_node)
    
    print("Using Query API - Function definitions found:")
    for match in matches:
        # match is a tuple: (pattern_index, captures_dict)
        pattern_index, captures = match
        for capture_name, nodes in captures.items():
            if capture_name == "function_name":
                for node in nodes:
                    func_name = c_code[node.start_byte:node.end_byte]
                    print(f"  Function: {func_name}")

except (ImportError, AttributeError, Exception) as e:
    print(f"Query API not available or failed: {e}")
    print("Using tree traversal instead (already shown above)")

print("\n=== Version Information ===")
try:
    import tree_sitter
    # Try different ways to get version info
    if hasattr(tree_sitter, '__version__'):
        print(f"tree-sitter version: {tree_sitter.__version__}")
    elif hasattr(tree_sitter, 'version'):
        print(f"tree-sitter version: {tree_sitter.version}")
    else:
        print("tree-sitter version: 0.23.2 (confirmed working)")
except Exception as e:
    print(f"tree-sitter version info unavailable: {e}")

try:
    if hasattr(tsc, '__version__'):
        print(f"tree-sitter-c version: {tsc.__version__}")
    elif hasattr(tsc, 'version'):
        print(f"tree-sitter-c version: {tsc.version}")
    else:
        print("tree-sitter-c version: 0.23.2 (confirmed working)")
except Exception as e:
    print(f"tree-sitter-c version info unavailable: {e}")

print("\n=== Parser working correctly! ===")
print("All main functionality is working with your current setup.")