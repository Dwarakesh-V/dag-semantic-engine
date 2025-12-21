# Built-in
from math import exp
from typing import List
from numpy import mean
from torch import tensor
from collections import deque
import json

# Downloaded
from sentence_transformers import SentenceTransformer, util
from numpy import array

# Custom
from node import node
from split_parse import split_parse
from retrieval_json_store import save_retrieval_record, load_retrieval_store

MODEL = SentenceTransformer("all-MiniLM-L12-v2")
HIGH_CONFIDENCE_THRESHOLD = 0.35
PASS_PREV_DATA = []
PREV_DATA = deque([None,None,None])
RAF_MULTIPLIER = 0.15
RETRIEVED_DATA = load_retrieval_store()

def retrieval_conf(enc_query, confidences, node):
    # Retrieval augmented fallback
    cur_max_confidence = 0
    curnode = None
    for entry in RETRIEVED_DATA:
        conf = util.cos_sim(tensor(entry["embedding"]).to(device="cuda"),enc_query).max().item()
        if conf > cur_max_confidence:
            curnode = entry["node_id"]
            cur_max_confidence = conf
    
    # If the retrieved node exists in the children of current node, add confidence to that child.
    for i in range(len(node.children)):
        if node.children[i].id == curnode:
            confidences[i] += RAF_MULTIPLIER*cur_max_confidence

    return confidences

def ct_query(query: str, base=0.6, decay=0.03, min_threshold=0.25)->float: # Confidence is a function based on length
    length = len(query.split())
    return max(base * exp(-decay * length), min_threshold)

def ct_depth(depth: int, base=0.2, decay=0.15, max_threshold=0.06) -> float: # Lower required confidence over depth
    return min(base * exp(-decay * depth), max_threshold)

def encode_query(query:str):
    return MODEL.encode(query,convert_to_tensor=True)

def sibling_pen(confs:List[int]):
    mean_pen = mean(confs)
    for i in range(len(confs)):
        confs[i]=confs[i]-mean_pen+confs[i]/len(confs)
    return confs

def single_level_pass(node:node,enc_query:tensor,query:str=""):
    val = [util.cos_sim(enc_query,child.example_embeddings).max().item() for child in node.children]

    # DEBUG
    # print("#######")
    # print(query)
    # print(all(MODEL.encode(query,convert_to_tensor=True)==enc_query))
    # print([child.examples for child in node.children])
    # print([util.cos_sim(enc_query,child.example_embeddings) for child in node.children])
    # print("#######")

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

def depth_pass(enc_query:tensor,query:str="",level=0):
    node = PASS_PREV_DATA[-1]
    print(f"Depth pass metadata\nRecursion level: {level}\nPrevious passes: {[node.intent for node in PASS_PREV_DATA]}\n--------------------\n")
    hct = ct_query(query) - ct_depth(level) # High confidence threshold
    lct = max(0,hct-0.2) # Low confidence threshold
    print(f"High confidence threshold: {hct} | Low confidence threshold: {lct}")
    confidences = single_level_pass(node,enc_query,query)

    if not confidences:
        PASS_PREV_DATA.pop()
        return

    # Rolling memory context - Top 3 historical intent matching
    confs = []
    for child in node.children:
        prev_enc = []
        for prev in PREV_DATA:
            if isinstance(prev,str):
                prev_enc.append(MODEL.encode(prev))
        prev_enc = array(prev_enc)
        if prev_enc.size>0:
            confs.append(util.cos_sim(prev_enc,MODEL.encode(child.intent)).max().item())
            best_idx = confs.index(max(confs))
            if max(confs) > 0.4:
                confidences[best_idx] += 0.1 * max(confs)
    # ---
    print(f"Updated confidences (Rolling memory): {confidences}")

    if not any(confidences) > lct:
        confidences = retrieval_conf(enc_query,confidences,node)
    if not any(confidences) > lct:
        if len(PASS_PREV_DATA)==1:
            print("Your query was unable to be processed. Please try rephrasing it.")
            return
        else:
            PASS_PREV_DATA.pop()
            depth_pass(enc_query,query,level+1)
            return

    if all(c < hct for c in confidences) and all(c >= lct for c in confidences):
        confidences = retrieval_conf(enc_query,confidences,node)

    # Now check if any confidence is higher than high confidence threshold - If not, ask user for data.
    if all(c < hct for c in confidences) and all(c >= lct for c in confidences):
        # Ask for clarification
        print(f"You are currently in \"{node.intent}\" for query {query}\nWhat are you looking for?")
        for child in node.children:
            print(f"{child.intent}")
        print([node.intent for node in PASS_PREV_DATA])
        user_clarify = input("You: ")
        user_encode = MODEL.encode(user_clarify,convert_to_tensor=True)
        print("ncall")
        new_confidences = single_level_pass(node,user_encode,user_clarify)
        while max(new_confidences) < HIGH_CONFIDENCE_THRESHOLD:
            max_confidence = max(new_confidences)
            if max_confidence < HIGH_CONFIDENCE_THRESHOLD:
                user_clarify = input("Please try again: ")
                user_encode = MODEL.encode(user_clarify,convert_to_tensor=True)
                new_confidences = single_level_pass(node,user_encode,user_clarify)

        # Storing for retrieval augmented fallback
        embedding_list = enc_query.detach().cpu().tolist() # Unpacking into values because JSON cannot serialize tensors
        raf_record = {
            "embedding": embedding_list,
            "node_id": node.children[new_confidences.index(max(new_confidences))].id,
            "original_query": query
        }
        save_retrieval_record(raf_record)
        depth_pass(user_encode,user_clarify,level+1)
        return

    for i in range(len(confidences)): # If multiple children have high confidences, pass it through all of them. If there are no children, then intent is completed.
        if confidences[i]>hct:
            PASS_PREV_DATA.append(node.children[i])
            PREV_DATA.append(node.children[i].intent) # Add string to PREV_DATA
            PREV_DATA.popleft()
            # Pass with that node that root if it has children. If it has no children, return that result.
            depth_pass(enc_query,query,level+1)
    
    PASS_PREV_DATA.pop()
    print(f"Completed \"{node.intent}\"")

def iter_pass(user_query:str,node:node):
    global PASS_PREV_DATA
    PASS_PREV_DATA.append(node)
    queries = split_parse(user_query)
    for query,_ in queries:
        enc_query = MODEL.encode(query,convert_to_tensor=True)
        depth_pass(enc_query,query)
