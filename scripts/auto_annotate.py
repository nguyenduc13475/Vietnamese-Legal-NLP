import json
import os
import sys

import stanza
import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize

# Import trained NER and Dep mappings from engine to ensure consistency
from src.extraction.ner_engine import extract_ultra_entities
from src.extraction.srl_engine import DEP_MAP, NER_MAP

TOKENIZER_NAME = "vinai/phobert-base"

# Initialize Stanza with pre-tokenization support
try:
    stanza_nlp = stanza.Pipeline(
        "vi",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=torch.cuda.is_available(),
    )
except Exception:
    stanza.download("vi")
    stanza_nlp = stanza.Pipeline(
        "vi", processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True
    )

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    print(f"Error loading PhoBERT tokenizer: {e}")
    sys.exit(1)


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
    return tokens, tags


def process_segmentation(segment_raw_path: str):
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
        tokens, tags = assign_labels_to_tokens(
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


def process_annotated_tasks(annotated_raw_path: str):
    with open(annotated_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    intent_data, ner_data, srl_data = [], [], []

    for item in raw_data:
        clause = item.get("clause", "")
        if not clause:
            continue

        # 1. Intent data
        intent_data.append({"text": clause, "label": item.get("intent", "Other")})

        # 2. Synchronized Tokenization
        raw_tokens = word_tokenize(clause)

        # 3. Structural Feature Extraction
        ner_results = extract_ultra_entities(clause)
        doc = stanza_nlp([raw_tokens])
        stanza_words = doc.sentences[0].words

        # 4. PhoBERT Subword Alignment
        encoding = tokenizer(
            raw_tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        word_ids = encoding.word_ids()
        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        sub_ner_ids = []
        sub_dep_ids = []
        sub_p_ner_ids = []

        for i, w_idx in enumerate(word_ids):
            if w_idx is None:
                sub_ner_ids.append(0)
                sub_dep_ids.append(0)
                sub_p_ner_ids.append(0)
                continue

            current_token_text = raw_tokens[w_idx]
            token_ner_label = "O"
            for ent in ner_results:
                if current_token_text in ent["text"]:
                    token_ner_label = ent["label"]
                    break
            sub_ner_ids.append(NER_MAP.get(token_ner_label, 0))

            sw = stanza_words[w_idx]
            sub_dep_ids.append(DEP_MAP.get(sw.deprel, 0))

            p_idx = sw.head - 1
            p_ner_label = "O"
            if p_idx >= 0:
                p_text = raw_tokens[p_idx]
                for ent in ner_results:
                    if p_text in ent["text"]:
                        p_ner_label = ent["label"]
                        break
            sub_p_ner_ids.append(NER_MAP.get(p_ner_label, 0))

        # 5. SRL Alignment using character offsets (Strict Alignment)
        srl_roles = item.get("srl_roles", [])
        srl_spans = extract_spans_sequentially(clause, srl_roles)
        sub_srl_tags = ["OTHER"] * len(input_ids)

        # Calculate character offsets for raw_tokens relative to original clause
        token_offsets = []
        cursor = 0
        for tok in raw_tokens:
            start = clause.find(tok, cursor)
            if start == -1:
                start = cursor  # Fallback
            end = start + len(tok)
            token_offsets.append((start, end))
            cursor = end

        for i, w_idx in enumerate(word_ids):
            if w_idx is None:
                continue
            # Find char range of this specific subword in original clause
            sub_start, sub_end = offset_mapping[i]
            word_start_in_clause = token_offsets[w_idx][0]

            actual_sub_start = word_start_in_clause + sub_start
            actual_sub_end = word_start_in_clause + sub_end

            for span in srl_spans:
                # If subword char range overlaps with annotated SRL span
                if actual_sub_start >= span["start"] and actual_sub_end <= span["end"]:
                    sub_srl_tags[i] = span["label"]
                    break

        ner_data.append({"input_ids": input_ids, "ner_tags": sub_ner_ids})
        srl_data.append(
            {
                "input_ids": input_ids,
                "srl_tags": sub_srl_tags,
                "ner_ids": sub_ner_ids,
                "dep_ids": sub_dep_ids,
                "p_ner_ids": sub_p_ner_ids,
            }
        )

    return intent_data, ner_data, srl_data


def split_and_save_all(segment_raw_path: str, annotated_raw_path: str, output_dir: str):
    if not os.path.exists(segment_raw_path) or not os.path.exists(annotated_raw_path):
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    segmentation_data = process_segmentation(segment_raw_path)
    intent_data, ner_data, srl_data = process_annotated_tasks(annotated_raw_path)

    def save(name, data):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    idx_s = int(len(segmentation_data) * 0.9)
    save("segment_train.json", segmentation_data[:idx_s])
    save("segment_test.json", segmentation_data[idx_s:])

    idx_i = int(len(intent_data) * 0.9)
    save("intent_train.json", intent_data[:idx_i])
    save("intent_test.json", intent_data[idx_i:])

    idx_n = int(len(ner_data) * 0.9)
    save("ner_train.json", ner_data[:idx_n])
    save("ner_test.json", ner_data[idx_n:])

    idx_sr = int(len(srl_data) * 0.9)
    save("srl_train.json", srl_data[:idx_sr])
    save("srl_test.json", srl_data[idx_sr:])


if __name__ == "__main__":
    split_and_save_all(
        "data/segment_raw.json", "data/annotated_raw.json", "data/annotated"
    )
