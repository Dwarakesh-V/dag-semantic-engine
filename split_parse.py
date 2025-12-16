# Built-in
import re

# Downloaded
from nltk import sent_tokenize

# Custom
from spacy_parse import extract_place_and_date

def split_parse(query):
    texts = sent_tokenize(query)
    res_queries = []
    p_queries = []
    for text in texts:
        split_text = [s.strip() for s in re.split(r'\b(and|then|also)\b', text) if s.strip() and s not in {"and", "then", "also"}]
        for i in split_text:
            res_queries.append(i)
    for query in res_queries:
        p_queries.append(extract_place_and_date(query))
    return p_queries

if __name__=="__main__":
    text = "Book me a flight on January 16th and Cancel my flight on December 28th. Book me a train on June 12th, 2026 to Banglore."
    print(split_parse(text))
