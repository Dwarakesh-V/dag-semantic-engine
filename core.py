from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from util import select_random_from_list
from math import exp
import sys

all_MiniLM_L12_v2 = SentenceTransformer("all-MiniLM-L12-v2")
"""
all-MiniLM-L12-v2 is a sentence embedding model used for tasks involving semantic textual similarity, clustering, semantic search, and information retrieval. They convert a sentence to tensor based on their intent and then matching patterns like cos_sim can be used to compare them to other sentences.
"""

CONNECTION_PHRASES = ["just like that", "for the same", "similarly", "similar to the previous", "for that", "for it"]
CONNECTION_ENCODE = all_MiniLM_L12_v2.encode(CONNECTION_PHRASES,convert_to_tensor=True)

CLEAR_MESSAGES = ["delete", "delete context", "delete history", "clear", "clear context", "clear history", "reset", "reset context", "reset chat", "forget", "forget all"]
CLEAR_MESSAGES_ENCODE = all_MiniLM_L12_v2.encode(CLEAR_MESSAGES,convert_to_tensor=True)

prev_label = ""
prev_query_data = [] # Stores previous context if queries contain ambiguous content that may map to previous responses

confidence_threshold = 0.35 # Default confidence threshold
def generate_confidence_threshold(query: str, base=0.6, decay=0.03, min_threshold=0.25)->float:
    """Generate confidence threshold based on the sentence. Longer sentence lead to lower confidences, so confidence threshold is adjusted based on that.
    Parameters:
    1. query: Modify threshold based on this sentence
    2. base, decay: 0.8*e^(-(decay * no. of words in string))
    3. min_threshold: Clamp to minimum to avoid much lower confidence values"""
    global confidence_threshold
    length = len(query.split())
    confidence_threshold = max(base * exp(-decay * length), min_threshold)

# The value of each node contains the following data
# node.value[0] -> intent
# node.value[1] -> label
# node.value[2] -> examples
# node.value[3] -> response
# node.value[4] -> children

def cache_embeddings(tree, model = all_MiniLM_L12_v2)->None:
    """Store the encoded examples as part of the tree itself to avoid repetitive computations.
    Parameters:
    1. tree: Tree to cache embeddings
    2. model: Which model to use to encode (Default: Global model all_MiniLM_L12_v2)"""

    def _cache_node_embeddings(n):
        if isinstance(n.value, tuple) and len(n.value) >= 2:
            examples = n.value[2]
            n.embedding_cache = model.encode(examples, convert_to_tensor=True)
        for child in n.children:
            _cache_node_embeddings(child)
    _cache_node_embeddings(tree.root)

SECOND_PERSON_MENTIONS = ["you", "youre", "your", "yours", "yourself", "y'all", "y'all's", "y'all'self", "you're", "your'e""u", "ur", "urs", "urself"]
def get_user_query(message="", model = all_MiniLM_L12_v2)->str:
    """Separate function to get input from user.
    Parameters:
    1. message: Show message to user before recieving input (Default: empty)
    2. model: Which model to use to encode (Default: Global model all_MiniLM_L12_v2)"""

    query = input(message).lower().strip()
    while query == "":
        query = input(message).lower().strip()
    
    query = query.replace("  "," ") # Remove double spaces
    for spm in SECOND_PERSON_MENTIONS: # Remove second person mentions
        query = query.replace(spm,"Amber AI")  # Replace with bot name
    generate_confidence_threshold(query)
    query_encode = model.encode(query, convert_to_tensor=True)
    clear_intent = util.cos_sim(query_encode,CLEAR_MESSAGES_ENCODE).max().item()
    if clear_intent > confidence_threshold:
        return None
    return query

def _calculate_single_level(user_embed,predicted_intent):
    """Calculate predictions for the children of a single node. Each node contains a list of nodes as its children.
    Parameters:
    1. user_embed: User query converted to tensor
    2. predicted_intent: Calculate for children of this node"""

    categories = predicted_intent.children # List of node objects
    predicted_intent = None
    high_intent = 0
    for category in categories:
        if category.embedding_cache is None:
            raise ValueError("Embedding cache missing. Call cache_embeddings() on the tree first")
        score = util.cos_sim(user_embed, category.embedding_cache).max().item()

        if score > high_intent:
            high_intent = score
            predicted_intent = category # Node object
    return (predicted_intent,high_intent) # Returns the child node with the highest prediction confidence and the confidence value

def _store_prev_data(predicted_intent):
    """Store the previous computed data path.
    Parameters:
    1. predicted_intent: Store previous data w.r.t this node"""
    # Mutating global prev_query_data
    prev_query_data.clear()
    prev_context_treenode = predicted_intent
    while prev_context_treenode.parent: # Stop at tree root
        prev_query_data.append(prev_context_treenode)
        prev_context_treenode = prev_context_treenode.parent

def h_pass(tree, user_embed, predicted_intent = None)->tuple:
    """Use the model to pass through the tree to compare it with the user query in a hierarchical manner and return an output.
    Parameters:
    1. tree: Which tree to pass through hierarchically
    2. user_embed: User input converted to tensor
    3. predicted_intent: Where to start the pass from (Default: Root of the tree)"""
    global prev_label
    predicted_intent = tree.root if predicted_intent == None else predicted_intent
    predicted_intent_parent = None
    high_intent = 0
    passed_once = False
    pass_through_intent = {}
    while predicted_intent.children: # If the node has children, check for the child with the highest confidence value
        predicted_intent_parent = predicted_intent
        predicted_intent, high_intent = _calculate_single_level(user_embed,predicted_intent)
        pass_through_intent[predicted_intent] = high_intent # Store the confidence value of the current node
        if passed_once: # If the data didn't pass even once, then don't store it
            _store_prev_data(predicted_intent_parent) # Storing previous data w.r.t parent node as context is changed from current node
        if high_intent < confidence_threshold: # If highest confidence value is still too low, stop.
            prev_label = predicted_intent_parent.value[1]
            return (predicted_intent, predicted_intent_parent, high_intent, passed_once, False, pass_through_intent) # If the confidence value is low, stop   
        passed_once = True
    
    _store_prev_data(predicted_intent)
    prev_label = predicted_intent.value[1]
    return (predicted_intent, predicted_intent_parent, high_intent, passed_once, True, pass_through_intent)

def query_pass(tree, user_input, model=all_MiniLM_L12_v2)->list:
    """Separate multiple queries into separate single ones, analyze relation between them if any, and process them to give an output while storing incomplete query outputs in non-leaf list, which contains the current level of context.
    Parameters:
    1. tree: Which tree to pass through hierarchically
    2. user_input: User input that may contain one or more queries as a string
    3. model: Which model to use to encode (Default: Global model all_MiniLM_L12_v2)"""

    queries = sent_tokenize(user_input)
    user_embeddings = [model.encode(query,convert_to_tensor=True) for query in queries]
    result = []
    label = prev_label

    for i in range(len(queries)):
        generate_confidence_threshold(queries[i])
        pass_value = (None, None, 0, False, False, None)
        # pass_value[0] -> current predicted intention (node)
        # pass_value[1] -> parent node of current predicted intention
        # pass_value[2] -> confidence level
        # pass_value[3] -> has the query passed through the model atleast once?
        # pass_value[4] -> has the query reached a leaf node?
        # pass_value[5] -> confidence values of traversal for query [DEBUGGING PURPOSES]

        # Acquiring data from previous query if the query has words matching with connecting phrases
        conn_sim = util.cos_sim(user_embeddings[i], CONNECTION_ENCODE).max().item()
        if conn_sim > confidence_threshold:
            queries[i] = queries[i] + label
            user_embeddings[i] = model.encode(queries[i], convert_to_tensor=True)

        # Pass values through the root node and the nodes that have the current context
        pass_value_root = h_pass(tree,user_embeddings[i]) # Passing through root node
        pass_value_nonleaf = [h_pass(tree,user_embeddings[i],j) for j in prev_query_data] # Passing through nodes that have current context
        all_nodes = [pass_value_root] + pass_value_nonleaf # List of all nodes that have been passed through
        pass_value = max(all_nodes, key=lambda x: x[2]) # Maximum confidence node for available context. Root is always a context.
        print(f"Query reach confidence: {[i[5] for i in all_nodes]}", file=sys.stderr) # DEBUGGING PURPOSES

        if pass_value[3]: # If the query has passed at least once, ask for data and store current result
            if not pass_value[4]: # If pass has not reached a leaf node, then ask for more data from the user and keep parent context
                label = pass_value[1].value[1]
                result.append(f"{pass_value[1].value[3]}")
                # continue

            else: # Query has reached a leaf node
                label = pass_value[0].value[1]
                result.append(pass_value[0].value[3])
                # continue

        else: # Query has not passed even once. Check if it works when previous context is available
            for parent_context in prev_query_data:
                pass_value_context = h_pass(tree, user_embeddings[i], parent_context)
                if pass_value_context[3]: # Check if it has passed at least once
                    # If it has passed, then the query is valid
                    if not pass_value_context[4]: # If pass has not reached a leaf node, then ask for more data from the user and keep parent context
                        label = pass_value_context[1].value[1]
                        result.append(f"What are you looking for in {pass_value_context[1].value[0]}? {pass_value_context[1].value[3]}")
                    else:
                        label = pass_value_context[0].value[1]
                        result.append(pass_value_context[0].value[3])
                    break # The else block won't be executed if code reaches here
            else: # The else statement of a for loop will execute only if the loop completes, and won't execute when broken by "break"
                result.append(f"I don't quite understand what you are trying to ask by \"{queries[i]}\"")
                # continue
    # End of "for" loop processing queries

    # Finally, return result. A list of responses same as the length of queries.
    return result

def process_user_query(query: str, model = all_MiniLM_L12_v2)->str:
    """Separate function to get input from user.
    Parameters:
    1. message: Show message to user before recieving input (Default: empty)
    2. model: Which model to use to encode (Default: Global model all_MiniLM_L12_v2)"""

    query = query.lower().strip()
    generate_confidence_threshold(query)
    query_encode = model.encode(query, convert_to_tensor=True)
    clear_intent = util.cos_sim(query_encode,CLEAR_MESSAGES_ENCODE).max().item()
    if clear_intent > confidence_threshold:
        return None
    return query

def interact_with_user(tree_data, user_input: str) -> str:
    """Handles a single user query and returns a response string."""
    user_input = process_user_query(user_input)
    all_results = []
    if user_input:  # If not empty or command
        results = query_pass(tree_data, user_input)
        for result in results:
            # return f"{select_random_from_list(result)}\nContext window: {prev_query_data}"
            all_results.append(f"{select_random_from_list(result)}")
        print(f"Previous query data: {prev_query_data}", file=sys.stderr)
        return all_results
    else:
        # Mutating global variables: Clearing context and recent history on command
        prev_query_data.clear()
        print(f"Previous query data: {prev_query_data}", file=sys.stderr)
        return ["Cleared previous context"]
    
    