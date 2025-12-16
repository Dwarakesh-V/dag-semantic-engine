import sys
import json
from core import interact_with_user, cache_embeddings
from json_parser import tree_from_json
from tree import struct_tree
from time import time
import os

# Load model once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
print(PROJECT_ROOT)

filepath_root = os.path.join(PROJECT_ROOT, "semantic-analyzer")
filepath_naive = "company_tree_data.json"
filepath = os.path.join(filepath_root, filepath_naive)

print("Loading engine...", file=sys.stderr)

s = time()
tree_file = tree_from_json(filepath)
print(f"Loaded tree json in {time() - s:.2f}s", file=sys.stderr)

s = time()
cache_embeddings(tree_file)
print(f"Cached embeddings in {time() - s:.2f}s", file=sys.stderr)

tree_file.save(filepath[:-5] + ".pkl")

s = time()
tree_data = struct_tree.load(filepath[:-5] + ".pkl")
print(f"Loaded tree pickle in {time() - s:.2f}s", file=sys.stderr)

print("READY", file=sys.stderr)
sys.stderr.flush()

# Request loop
while True:
    line = input()
    if not line:
        break

    try:
        # data = json.loads(line)
        # user_query = data.get("query", "")
        result = interact_with_user(tree_data, line)
        print(json.dumps({"response": result}) + "\n")
    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
        sys.stdout.flush()
