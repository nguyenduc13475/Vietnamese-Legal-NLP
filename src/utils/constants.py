# Constants for structural mapping to ensure consistency across training and inference
NER_LABEL_MAP = {
    "O": 0,
    "B-PARTY": 1,
    "I-PARTY": 2,
    "B-MONEY": 3,
    "I-MONEY": 4,
    "B-DATE": 5,
    "I-DATE": 6,
    "B-RATE": 7,
    "I-RATE": 8,
    "B-PENALTY": 9,
    "I-PENALTY": 10,
    "B-LAW": 11,
    "I-LAW": 12,
    "B-OBJECT": 13,
    "I-OBJECT": 14,
    "B-PREDICATE": 15,
    "I-PREDICATE": 16,
}

# Simplified map for SRL/Internal features (maps B/I to base category)
NER_CATEGORY_MAP = {
    "O": 0,
    "PARTY": 1,
    "MONEY": 2,
    "DATE": 3,
    "RATE": 4,
    "PENALTY": 5,
    "LAW": 6,
    "OBJECT": 7,
    "PREDICATE": 8,
}

DEP_RELATION_MAP = {
    "root": 1,
    "nsubj": 2,
    "obj": 3,
    "iobj": 4,
    "obl": 5,
    "advcl": 6,
    "amod": 7,
    "nmod": 8,
    "compound": 9,
    "mark": 10,
    "advmod": 11,
    "xcomp": 12,
    "cc": 13,
    "conj": 14,
    "det": 15,
    "case": 16,
    "fixed": 17,
    "flat": 18,
    "punct": 19,
}

SRL_ROLE_MAP = {
    "OTHER": 0,
    "AGENT": 1,
    "RECIPIENT": 2,
    "THEME": 3,
    "NAME": 4,
    "TIME": 5,
    "CONDITION": 6,
    "TRAIT": 7,
    "LOCATION": 8,
    "METHOD": 9,
    "ABOUT": 10,
}
