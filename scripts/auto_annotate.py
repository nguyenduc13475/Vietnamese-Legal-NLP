import json
import os
import sys

import stanza
from tqdm import tqdm
from transformers import AutoTokenizer
from underthesea import word_tokenize

# Constants for structural mapping (must match srl_engine.py)
NER_MAP = {
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

DEP_MAP = {
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

SRL_MAP = {
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

TOKENIZER_NAME = "vinai/phobert-base"

# Initialize Tools
print("--- Loading Stanza and Tokenizer ---")
try:
    # Use pre-tokenized mode for exact word matching
    nlp = stanza.Pipeline("vi", processors="tokenize,pos,lemma,depparse", verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    print(f"Init Error: {e}")
    sys.exit(1)


# --- UTILITY FUNCTIONS ---


def get_word_ner_label(word_start, word_end, ner_spans, clause_text):
    """Map a Stanza word span to a NER label from annotated JSON."""
    for span in ner_spans:
        s_text = span.get("text", "")
        s_start = clause_text.find(s_text)
        if s_start == -1:
            continue
        s_end = s_start + len(s_text)

        if word_start >= s_start and word_end <= s_end:
            return span.get("label", "O")
    return "O"


def extract_spans_sequentially(clause: str, components: list) -> list:
    spans = []
    search_start_idx = 0

    for comp in components:
        substr = comp.get("text", "").strip()
        label = comp.get("label")

        if not substr:
            continue

        start_char = clause.find(substr, search_start_idx)

        if start_char == -1:
            start_char = clause.find(substr)
            if start_char == -1:
                continue

        end_char = start_char + len(substr)
        search_start_idx = end_char

        spans.append(
            {"text": substr, "label": label, "start": start_char, "end": end_char}
        )

    return spans


def assign_labels_to_tokens(clause: str, spans: list, default_label, use_bio=False):
    segmented_clause = word_tokenize(clause, format="text")

    encoding = tokenizer(segmented_clause, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    tags = [default_label] * len(tokens)

    clause_stripped = "".join(clause.split()).lower()
    token_spans_in_stripped = []
    current_pos = 0

    for token in tokens:
        clean_tok = token.replace("@@", "").replace("_", "").lower()
        if not clean_tok:
            token_spans_in_stripped.append((-1, -1))
            continue

        start = clause_stripped.find(clean_tok, current_pos)
        if start != -1:
            end = start + len(clean_tok)
            token_spans_in_stripped.append((start, end))
            current_pos = end
        else:
            token_spans_in_stripped.append((-1, -1))

    for span in spans:
        prefix = clause[: span["start"]]
        prefix_stripped_len = len("".join(prefix.split()))
        content_stripped_len = len("".join(span["text"].split()))

        stripped_start = prefix_stripped_len
        stripped_end = stripped_start + content_stripped_len

        label = span["label"]
        is_first_token_in_span = True

        for i, (tok_s, tok_e) in enumerate(token_spans_in_stripped):
            if tok_s == -1:
                continue

            if tok_s >= stripped_start and tok_e <= stripped_end:
                if use_bio:
                    tags[i] = f"B-{label}" if is_first_token_in_span else f"I-{label}"
                    is_first_token_in_span = False
                else:
                    tags[i] = label

    if use_bio:
        for i in range(1, len(tags)):
            if tags[i].startswith("I-"):
                prev_tag = tags[i - 1]
                target_label = tags[i].split("-")[1]
                if prev_tag == "O" or (
                    prev_tag != f"B-{target_label}" and prev_tag != f"I-{target_label}"
                ):
                    tags[i] = f"B-{target_label}"

    return tokens, tags, token_spans_in_stripped


# --- DATA PROCESSING FUNCTIONS ---


def process_segmentation(segment_raw_path: str):
    """Process Segmentation dataset using standard matching logic."""
    print(f"📖 Processing Segmentation from {segment_raw_path}...")
    with open(segment_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = []
    for item in raw_data:
        clause = item.get("clause", "")
        segments = item.get("segment", [])
        if not clause or not segments:
            continue

        components = []
        for idx, seg_text in enumerate(segments):
            if seg_text.strip():
                components.append({"text": seg_text, "label": idx + 1})

        spans = extract_spans_sequentially(clause, components)

        tokens, tags, _ = assign_labels_to_tokens(
            clause, spans, default_label=0, use_bio=False
        )

        token_ids = [
            tokenizer.convert_tokens_to_ids(t)
            if tokenizer.convert_tokens_to_ids(t) is not None
            else tokenizer.unk_token_id
            for t in tokens
        ]

        dataset.append({"input_ids": token_ids, "segment_tags": tags})

    return dataset


def process_intent_and_ner(annotated_raw_path: str):
    """Process Intent and NER datasets."""
    print(f"📖 Processing Intent and NER from {annotated_raw_path}...")
    with open(annotated_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    intent_data, ner_data = [], []

    for item in raw_data:
        clause = item.get("clause", "")
        if not clause:
            continue

        intent_data.append({"text": clause, "label": item.get("intent", "Other")})

        ultra_ner = item.get("ultra_ner", [])
        srl_roles = item.get("srl_roles", [])

        srl_spans = extract_spans_sequentially(clause, srl_roles)
        ner_spans = extract_spans_sequentially(clause, ultra_ner)

        # Merge predicate from SRL roles into NER as per original logic
        predicate_spans = [span for span in srl_spans if span["label"] == "PREDICATE"]

        combined_ner_spans = ner_spans + predicate_spans
        combined_ner_spans.sort(key=lambda x: x["start"])

        ner_tokens, ner_tags, _ = assign_labels_to_tokens(
            clause, combined_ner_spans, default_label="O", use_bio=True
        )

        token_ids = [
            tokenizer.convert_tokens_to_ids(t)
            if tokenizer.convert_tokens_to_ids(t) is not None
            else tokenizer.unk_token_id
            for t in ner_tokens
        ]

        ner_data.append({"input_ids": token_ids, "ner_tags": ner_tags})

    return intent_data, ner_data


def process_srl_data(annotated_raw_path):
    """Process SRL dataset ensuring precise token-alignment with NER."""
    print(f"📖 Processing SRL from {annotated_raw_path}...")
    with open(annotated_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    srl_dataset = []
    print(f"Generating SRL features for {len(raw_data)} clauses...")

    for item in tqdm(raw_data):
        clause = item.get("clause", "")
        if not clause:
            continue

        # 1. Dependency Analysis via Stanza (Solely for structure, NOT tokenization)
        doc = nlp(clause)
        if not doc.sentences:
            continue
        sentence = doc.sentences[0]
        stanza_words = sentence.words

        # 2. Extract strictly identical tokenization layout to NER
        srl_roles_raw = item.get("srl_roles", [])
        srl_spans = extract_spans_sequentially(clause, srl_roles_raw)

        tokens, srl_tags, token_spans = assign_labels_to_tokens(
            clause, srl_spans, default_label="OTHER", use_bio=False
        )

        # Calculate matching NER context natively
        ultra_ner = item.get("ultra_ner", [])
        ner_spans = extract_spans_sequentially(clause, ultra_ner)
        predicate_spans = [span for span in srl_spans if span["label"] == "PREDICATE"]
        combined_ner_spans = ner_spans + predicate_spans
        combined_ner_spans.sort(key=lambda x: x["start"])

        _, ner_tags, _ = assign_labels_to_tokens(
            clause, combined_ner_spans, default_label="O", use_bio=False
        )

        input_ids = [
            tokenizer.convert_tokens_to_ids(t)
            if tokenizer.convert_tokens_to_ids(t) is not None
            else tokenizer.unk_token_id
            for t in tokens
        ]

        # 3. Align Stanza Dependencies to our Tokenized Output via overlap
        clause_stripped = "".join(clause.split()).lower()
        stanza_spans = []
        current_pos = 0
        for w in stanza_words:
            clean_w = w.text.replace(" ", "").lower()
            start = clause_stripped.find(clean_w, current_pos)
            if start != -1:
                end = start + len(clean_w)
                stanza_spans.append((start, end))
                current_pos = end
            else:
                stanza_spans.append((-1, -1))

        # Build Parents Context Array
        stanza_ner_labels = []
        for w in stanza_words:
            label = get_word_ner_label(w.start_char, w.end_char, ultra_ner, clause)
            stanza_ner_labels.append(label)

        num_tokens = len(input_ids)
        ner_ids = [0] * num_tokens
        dep_ids = [0] * num_tokens
        p_ner_ids = [0] * num_tokens

        for i, (tok_s, tok_e) in enumerate(token_spans):
            # Resolve structural NER map
            clean_tag = ner_tags[i].replace("B-", "").replace("I-", "")
            ner_ids[i] = NER_MAP.get(clean_tag, 0)

            if tok_s == -1:
                continue

            # Determine maximum character-level overlap with a Stanza word
            best_w_idx = -1
            max_overlap = 0

            for w_idx, (w_s, w_e) in enumerate(stanza_spans):
                if w_s == -1:
                    continue
                overlap_start = max(tok_s, w_s)
                overlap_end = min(tok_e, w_e)
                overlap = overlap_end - overlap_start

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_w_idx = w_idx

            # Assign Dependency based on Maximum Stanza overlap match
            if best_w_idx != -1:
                current_word = stanza_words[best_w_idx]
                dep_ids[i] = DEP_MAP.get(current_word.deprel, 0)

                head_idx = current_word.head - 1
                if 0 <= head_idx < len(stanza_ner_labels):
                    p_ner_ids[i] = NER_MAP.get(stanza_ner_labels[head_idx], 0)

        srl_dataset.append(
            {
                "input_ids": input_ids,
                "srl_tags": srl_tags,
                "ner_ids": ner_ids,
                "dep_ids": dep_ids,
                "p_ner_ids": p_ner_ids,
            }
        )

    return srl_dataset


# --- MAIN PIPELINE ---


def split_and_save_all(segment_raw_path: str, annotated_raw_path: str, output_dir: str):
    if not os.path.exists(segment_raw_path):
        print(f"Error: Could not find {segment_raw_path}")
        sys.exit(1)
    if not os.path.exists(annotated_raw_path):
        print(f"Error: Could not find {annotated_raw_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Generate all datasets using respective logic
    segmentation_data = process_segmentation(segment_raw_path)
    intent_data, ner_data = process_intent_and_ner(annotated_raw_path)
    srl_data = process_srl_data(annotated_raw_path)

    # 90/10 Splits
    datasets = {
        "segment_train.json": segmentation_data[: int(len(segmentation_data) * 0.9)],
        "segment_test.json": segmentation_data[int(len(segmentation_data) * 0.9) :],
        "intent_train.json": intent_data[: int(len(intent_data) * 0.9)],
        "intent_test.json": intent_data[int(len(intent_data) * 0.9) :],
        "ner_train.json": ner_data[: int(len(ner_data) * 0.9)],
        "ner_test.json": ner_data[int(len(ner_data) * 0.9) :],
        "srl_train.json": srl_data[: int(len(srl_data) * 0.9)],
        "srl_test.json": srl_data[int(len(srl_data) * 0.9) :],
    }

    for filename, data in datasets.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved: {filepath}")

    print("\n✅ Success! All datasets generated with perfectly aligned token logic.")


if __name__ == "__main__":
    SEGMENT_RAW_JSON = "data/segment_raw.json"
    ANNOTATED_RAW_JSON = "data/annotated_raw.json"
    OUTPUT_DIRECTORY = "data/annotated"

    split_and_save_all(SEGMENT_RAW_JSON, ANNOTATED_RAW_JSON, OUTPUT_DIRECTORY)
