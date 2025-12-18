# Built-in
import json
from os.path import exists

# Downloaded
from sentence_transformers import SentenceTransformer

# Custom
import cache_graph
from json_parser import build_tree
from core_pass import iter_pass

FILENAME_BASE = "nodes.json"
FILENAME_CACHED = "nodes.pkl"
MODEL = SentenceTransformer("all-MiniLM-L12-v2")

if not exists(FILENAME_CACHED):
    if not exists(FILENAME_BASE):
        print(FILENAME_BASE, "does not exist.")
    else:
        with open(FILENAME_BASE) as tree_data:
            json_content = json.load(tree_data)
        root_node = build_tree(json_content)
        cache_graph.cache_embeddings(root_node,MODEL)
        cache_graph.save_graph_cache(root_node,FILENAME_CACHED)

root_node = cache_graph.load_graph_cache(FILENAME_CACHED)

if __name__ == "__main__":
    print("\nREADY\n")
    iter_pass(input("You: "),root_node)