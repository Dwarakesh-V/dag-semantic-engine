import json
import os

RAF_FILE = "retrieval_store.json"

def save_retrieval_record(record):
    if os.path.exists(RAF_FILE): # If file exists, add to that file. Otherwise create a new file
        with open(RAF_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"retrieval_data": []}

    data["retrieval_data"].append(record)

    with open(RAF_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_retrieval_store():
    if not os.path.exists(RAF_FILE):
        return []
    with open(RAF_FILE, "r") as f:
        return json.load(f)["retrieval_data"]
