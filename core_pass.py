# Built-in
from math import exp
from typing import List
from numpy import mean
from torch import tensor

# Downloaded
from sentence_transformers import SentenceTransformer, util

# Custom
from node import node

MODEL = SentenceTransformer("all-MiniLM-L12-v2")

def ct_query(query: str, base=0.6, decay=0.03, min_threshold=0.25)->float: # Confidence is a function based on length
    length = len(query.split())
    return max(base * exp(-decay * length), min_threshold)

def ct_depth(depth: int, base=0.6, decay=0.15, min_threshold=0.2) -> float: # Lower required confidence over depth
    return max(base * exp(-decay * depth), min_threshold)

def encode_query(query:str):
    return MODEL.encode(query,convert_to_tensor=True)

def sibling_pen(confs:List[int]):
    mean_pen = mean(confs)
    for i in range(len(confs)):
        confs[i]=confs[i]-mean_pen+confs[i]/len(confs)
    return confs

def single_level_pass(node:node,enc_query:tensor,query:str=""):
    val = [util.cos_sim(enc_query,child.example_embeddings).max().item() for child in node.children]
    for i in range(len(node.children)):
        if node.children[i].sp:
            val[i]=val[i]-mean(val)+val[i]/len(val)
    
    # DEBUG
    print("---single pass---")
    print(f"Query: {query}")
    print(f"Start node: {node.intent}")
    for c in range(len(node.children)):
        print(f"{node.children[c].intent}: {val[c]}")
    print("---end of single pass debug---\n")

    return val

def depth_pass(node:node,)
# Book me a flight on January 16th and Cancel my flight on December 28th. Book me a train on June 12th, 2026 to Banglore.
# [['Book me a flight on DATE', ('January 16th',)], ['Cancel my flight on DATE.', ('December 28th',)], ['Book me a train on DATE to PLACE.', ('June 12th, 2026', 'Banglore')]]