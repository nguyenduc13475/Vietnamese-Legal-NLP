def extract_srl(
    text: str, entities: list, dependencies: list = None, np_chunks: list = None
) -> dict:
    """
    Advanced SRL analysis combining NER, Dependency Parsing, and NP Chunking.
    """
    roles = {}
    predicate = "N/A"

    if dependencies:
        root_token = next(
            (d for d in dependencies if d.get("relation") == "root"), None
        )
        if root_token:
            predicate = root_token.get("token")
            root_idx = root_token.get("id")

            # Merge auxiliary/passive verbs and NEGATION WORDS
            aux_relations = ["pass", "aux", "aux:pass", "cop", "advmod"]
            aux_nodes = [
                (d.get("id"), d.get("token"))
                for d in dependencies
                if d.get("head_index") == root_idx
                and d.get("relation") in aux_relations
                and (
                    d.get("relation") != "advmod"
                    or d.get("token").lower()
                    in ["không", "chưa", "chẳng", "đừng", "tuyệt đối không"]
                )
            ]

            if aux_nodes:
                aux_sorted = sorted(aux_nodes, key=lambda x: x[0])
                aux_str = " ".join([t for idx, t in aux_sorted])
                predicate = f"{aux_str} {predicate}"

            # Capture the verb "chịu" alongside "có"
            if root_token.get("token").lower() in ["có", "chịu"]:
                comp_parts_tokens = [
                    d.get("token").lower()
                    for d in dependencies
                    if d.get("head_index") == root_idx
                ]
                text_lower = text.lower()

                for part in ["nghĩa vụ", "trách nhiệm", "quyền lợi", "quyền"]:
                    if f"{root_token.get('token').lower()} {part}" in text_lower or any(
                        p in part for p in comp_parts_tokens
                    ):
                        main_verb = next(
                            (
                                d.get("token")
                                for d in dependencies
                                if d.get("relation")
                                in ["xcomp", "ccomp", "acl", "vmod"]
                            ),
                            "",
                        )
                        if main_verb:
                            predicate = f"{predicate} {part} {main_verb}"
                        else:
                            predicate = f"{predicate} {part}"
                        break

    # Extract role candidates from syntax
    syntax_agents = (
        [d["token"] for d in dependencies if "nsubj" in d.get("relation", "")]
        if dependencies
        else []
    )
    syntax_themes = (
        [
            d["token"]
            for d in dependencies
            if d.get("relation") in ["obj", "iobj", "nsubj:pass"]
        ]
        if dependencies
        else []
    )

    # Combine Entities (NER) for semantic labeling
    for ent in entities:
        txt, lbl = ent["text"], ent["label"]
        if lbl == "PARTY":
            txt_lower = txt.lower()
            # Determine if this PARTY is likely an Agent (actor) or not
            is_agent = False

            # Match with the Agent list found from Dependency Parsing
            if any(agent.lower() in txt_lower for agent in syntax_agents):
                is_agent = True
            elif not syntax_agents and "Agent" not in roles:
                # If no Agent can be identified via syntax,
                # default to assigning the first PARTY as the Agent
                is_agent = True

            if is_agent and "Agent" not in roles:
                roles["Agent"] = txt
            elif "Recipient" not in roles:
                roles["Recipient"] = txt
            else:
                roles["Co-Party"] = txt
        elif lbl == "MONEY":
            roles["Theme"] = txt
        elif lbl == "DATE":
            roles["Time"] = txt
        elif lbl in ["RATE", "PENALTY"]:
            roles["Penalty_Rate"] = txt

    # Use NP Chunking to supplement the Theme if NER misses it
    if "Theme" not in roles and np_chunks:
        current_np = []
        extracted_nps = []
        for word, tag in np_chunks:
            if tag != "O":
                current_np.append(word)
            else:
                if current_np:
                    extracted_nps.append(" ".join(current_np))
                current_np = []
        if current_np:
            extracted_nps.append(" ".join(current_np))

        for np_str in extracted_nps:
            if syntax_themes:
                if any(st.lower() in np_str.lower() for st in syntax_themes):
                    roles["Theme"] = np_str
                    break
            else:
                is_used = False
                for existing_role in roles.values():
                    if (
                        existing_role.lower() in np_str.lower()
                        or np_str.lower() in existing_role.lower()
                    ):
                        is_used = True
                        break
                if not is_used and predicate.lower() not in np_str.lower():
                    roles["Theme"] = np_str
                    break

    # Extract Condition & Purpose recursively
    if dependencies:

        def get_subtree(node_id):
            """Hàm đệ quy gom toàn bộ nhánh cây phụ thuộc"""
            nodes = [d for d in dependencies if d.get("id") == node_id]
            children = [d for d in dependencies if d.get("head_index") == node_id]
            for child in children:
                nodes.extend(get_subtree(child.get("id")))
            return nodes

        for d in dependencies:
            if d.get("relation") == "mark":
                token_lower = d.get("token").lower()
                head_verb_idx = d.get("head_index")

                # Aggregate the entire head verb tree and reorder by ID to reconstruct the complete sentence
                clause_nodes = get_subtree(head_verb_idx)
                clause_nodes.sort(key=lambda x: x.get("id", 0))
                clause_text = " ".join([node["token"] for node in clause_nodes]).strip()

                if token_lower in [
                    "nếu",
                    "khi",
                    "trong trường hợp",
                    "trừ khi",
                    "giả sử",
                ]:
                    if "Condition" not in roles:
                        roles["Condition"] = clause_text
                elif token_lower in ["để", "nhằm", "vì"]:
                    if "Purpose" not in roles:
                        roles["Purpose"] = clause_text

    # Passive Voice Correction
    if any(
        passive_kw in predicate.lower() for passive_kw in ["bị", "được", "chịu phạt"]
    ):
        if "Agent" in roles and "Theme" not in roles:
            roles["Theme"] = roles.pop("Agent")
        elif "Agent" in roles and "Theme" in roles:
            roles["Agent"], roles["Theme"] = roles["Theme"], roles["Agent"]

    return {
        "predicate": predicate.strip(),
        "roles": {k: v for k, v in roles.items() if v},
    }
