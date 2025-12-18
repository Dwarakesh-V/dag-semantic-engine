# Built-in
from math import exp
from typing import List
from numpy import mean
from torch import tensor

# Downloaded
from sentence_transformers import SentenceTransformer, util

# Custom
from node import node
from split_parse import split_parse

MODEL = SentenceTransformer("all-MiniLM-L12-v2")
HIGH_CONFIDENCE_THRESHOLD = 0.35

def ct_query(query: str, base=0.5, decay=0.03, min_threshold=0.25)->float: # Confidence is a function based on length
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
    print("Single level pass--------------------")
    print(f"Query: {query}")
    print(f"Start node: {node.intent}")
    for c in range(len(node.children)):
        print(f"{node.children[c].intent}: {val[c]}")
    print("--------------------\n")

    return val

GLOBAL_PREV_DATA = []

def depth_pass(enc_query:tensor,query:str="",level=0):
    node = GLOBAL_PREV_DATA[-1]
    print(f"Depth pass metadata\nRecursion level: {level}\nPrevious passes: {[node.intent for node in GLOBAL_PREV_DATA]}\n--------------------\n")
    hct = ct_query(query) # High confidence threshold
    lct = max(0,hct-0.2) # Low confidence threshold
    print(f"High confidence threshold: {hct} | Low confidence threshold: {lct}")
    confidences = single_level_pass(node,enc_query,query)

    if not any(confidences) > lct:
        if len(GLOBAL_PREV_DATA)==1:
            print("Your query was unable to be processed. Please try rephrasing it.")
            return
        else:
            GLOBAL_PREV_DATA.pop(-1)
            depth_pass(enc_query,query)
            return
    
    if all(confidences) < hct and all(confidences) >= lct:
        # Ask for clarification
        print(f"You are currently in \"{node.intent}\" for query {query}\nWhat are you looking for?")
        for child in node.children:
            print(f"{child.intent}")
        user_clarify = input("You: ")
        user_encode = MODEL.encode(user_clarify,convert_to_tensor=True)
        depth_pass(node, user_encode, user_clarify,level+1)
        return

    for i in range(len(confidences)): # If multiple children have high confidences, pass it through all of them. If there are no children, then intent is completed.
        if confidences[i]>hct:
            GLOBAL_PREV_DATA.append(node.children[i])
            # Pass with that node that root if it has children. If it has no children, return that result.
            depth_pass(enc_query,query,level+1)
    else:
        print(f"Completed \"{node.intent}\"")

def iter_pass(user_query:str,node:node):
    global GLOBAL_PREV_DATA
    GLOBAL_PREV_DATA.append(node)
    queries = split_parse(user_query)
    for query,_ in queries:
        enc_query = MODEL.encode(query,convert_to_tensor=True)
        depth_pass(enc_query,query)

# Output of split_parse
# Book me a flight on January 16th and Cancel my flight on December 28th. Book me a train on June 12th, 2026 to Banglore.
# [['Book me a flight on DATE', ('January 16th',)], ['Cancel my flight on DATE.', ('December 28th',)], ['Book me a train on DATE to PLACE.', ('June 12th, 2026', 'Banglore')]]