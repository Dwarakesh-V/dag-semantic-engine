import pickle

class node:
    """Individual node class of each node of tree"""
    def __init__(self, value=None):
        self.value = value
        self.parent = None
        self.children = []
        self.embedding_cache = None  # Will hold precomputed embeddings if available
    
    def __repr__(self):
        return str(self.value[0]) if isinstance(self.value, tuple) else str(self.value)

class struct_tree:
    def __init__(self,value): # Root is mandatory
        self.root = node(value)
    
    def add_node(self, parent_node: node, value): # Add a node
        """Add node to the tree."""
        new_node = node(value)
        new_node.parent = parent_node
        parent_node.children.append(new_node)
        return new_node
    
    def visualize(self, show_labels=True):
        """Display the tree in a graph-like format."""        
        def _print_node(current_node, prefix="", is_last=True):
            # Print current node
            branch = "└── " if is_last else "├── "
            print(f"{prefix}{branch}{current_node.value if show_labels else 'o'}")
            
            # Prepare prefix for children
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
            
            # Print children
            child_count = len(current_node.children)
            for i, child in enumerate(current_node.children):
                is_last_child = i == child_count - 1
                _print_node(child, new_prefix, is_last_child)
        
        # Start recursive printing from root
        print(f"{self.root.value if show_labels else 'o'}")
        child_count = len(self.root.children)
        for i, child in enumerate(self.root.children):
            is_last_child = i == child_count - 1
            _print_node(child, "", is_last_child)
    
    def save(self, filepath: str):
        """Save the entire tree structure to disk with optional compression."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod # This method is called directly on the class rather than an instance of it
    def load(filepath: str):
        """Load the tree structure from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__": # Execute this only in this file
    # Create a tree with root value "A"
    tree = struct_tree("A")
    
    # Add some nodes
    b_node = tree.add_node(tree.root, "B")
    c_node = tree.add_node(tree.root, "C")
    d_node = tree.add_node(tree.root, "D")
    
    # Add children to B
    tree.add_node(b_node, "B1")
    b2_node = tree.add_node(b_node, "B2")
    tree.add_node(b2_node, "B2.1")
    
    # Add children to C
    tree.add_node(c_node, "C1")
    
    # Visualize the tree
    tree.visualize(show_labels=False)