import spacy
nlp = spacy.load("en_core_web_sm")
def extract_place_and_date(text: str)->list: # returns a list of length 2. Index 0: Parsed query. Index 1: Parsed values in a tuple
    doc = nlp(text)
    
    # create a list of entities with their positions
    entities = []
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):
            entities.append((ent.start_char, ent.end_char, "PLACE", ent.text))
        elif ent.label_ in ("DATE", "TIME"):
            entities.append((ent.start_char, ent.end_char, "DATE", ent.text))
    
    # sort entities by their start position
    entities.sort()
    
    # replace entities in the text with their labels (reverse order for replacement)
    modified_text = text
    for start, end, label, _ in reversed(entities):
        modified_text = modified_text[:start] + label + modified_text[end:]
    
    # extract values in order of appearance
    ordered_values = tuple(original_text for _, _, _, original_text in entities)
    
    return [modified_text, ordered_values]

if __name__=="__main__":
    # example
    print(extract_place_and_date("I visited Paris on June 12 in the year 2023 and moved to Berlin last year."))
    print(extract_place_and_date("I reached India yesterday."))
