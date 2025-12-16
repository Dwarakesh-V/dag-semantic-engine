# Built-in
import pickle

# Downloaded
from sentence_transformers import SentenceTransformer

# Custom
from node import node

def cache_embeddings(root: node, model: SentenceTransformer):
    stack = [root]
    while stack:
        current = stack.pop()
        if current.examples:
            current.example_embeddings = model.encode(
                current.examples,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        stack.extend(current.children)

def save_graph_cache(root: node, path: str):
    with open(path, "wb") as f:
        pickle.dump(root, f)

def load_graph_cache(path: str) -> node:
    with open(path, "rb") as f:
        return pickle.load(f)
