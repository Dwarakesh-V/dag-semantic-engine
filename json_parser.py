# Built-in
from typing import Dict

# Custom
from node import node

def build_tree(json_data: Dict) -> node:
    nodes: Dict[str, node] = {}

    # create nodes
    for key, data in json_data.items():
        n = node(
            idx=data["id"],
            intent=data["intent"],
            examples=data.get("examples", [])
        )
        n.sp = data.get("sibling_penalty", False)
        nodes[key] = n

    # link children
    for key, data in json_data.items():
        parent = nodes[key]
        for child_key in data.get("children", []):
            parent.children.append(nodes[child_key])

    # default root name
    return nodes["root"]
