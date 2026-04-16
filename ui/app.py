import ast

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Legal NLP Workspace", layout="wide", page_icon="⚖️")

# --- CUSTOM CSS FOR ENTERPRISE LOOK ---
st.markdown(
    """
    <style>
    .stApp header {background-color: transparent;}
    .iob-tag {padding: 3px 6px; border-radius: 4px; font-size: 0.85em; margin: 2px; display: inline-block;}
    .bio-tag {padding: 3px 6px; border-radius: 4px; font-size: 0.85em; margin: 2px; border: 1px solid #ccc;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- UTILS ---
@st.cache_data()
def fetch_sources():
    try:
        # Connect to the correct endpoint provided by the LegalRetriever
        res = requests.get(f"{API_URL}/database/sources")
        if res.status_code == 200:
            return res.json().get("sources", [])
    except Exception:
        pass
    return []


def render_nlp_dashboard(clause_data):
    """Visualizer đẹp mắt cho các yêu cầu của Bài tập lớn"""
    for item in clause_data:
        st.markdown(f'#### 🖋️ Mệnh đề: *"{item["clause"]}"*')

        # Assignment 2.3: Intent
        st.info(f"**Ý định (Intent):** {item['intent']}")

        # Assignment 1.2: NP Chunking (IOB Visualization)
        st.write("**[Tác vụ 1.2] Noun Phrase Chunking (Định dạng IOB)**")
        iob_html = ""
        for word, tag in item["np_chunks"]:
            color = "#e1f5fe" if "NP" in tag else "#f8f9fa"
            border = "2px solid #03a9f4" if "B-NP" in tag else "1px solid #dee2e6"
            iob_html += f"<div class='iob-tag' style='background:{color}; border:{border};'><b>{word}</b><br><small style='color:#6c757d'>{tag}</small></div>"
        st.markdown(iob_html, unsafe_allow_html=True)

        # Assignment 2.1: NER (BIO Visualization)
        st.write("**[Tác vụ 2.1] Named Entity Recognition (Định dạng BIO)**")
        import re

        bio_html = ""

        # Use the exact same tokenizer from auto_annotate.py to separate punctuation
        tokens = re.findall(r"\w+|[^\w\s]", item["clause"])
        tags = ["O"] * len(tokens)

        # Left-to-right strict matching pointer
        search_start_idx = 0

        # Check if entities is a string (from DB) or list (from direct API)
        current_entities = item["entities"]
        if isinstance(current_entities, str):
            try:
                current_entities = ast.literal_eval(current_entities)
            except Exception as e:
                print(e)
                current_entities = []

        for ent in current_entities:
            # Handle both dictionary objects and fallback strings
            if isinstance(ent, dict):
                ent_text = ent.get("text", "")
                ent_label = ent.get("label", "ENTITY")
            else:
                ent_text = str(ent)
                ent_label = "ENTITY"

            ent_tokens = re.findall(r"\w+|[^\w\s]", ent_text)
            ent_len = len(ent_tokens)

            if ent_len == 0:
                continue

            for i in range(search_start_idx, len(tokens) - ent_len + 1):
                if tokens[i : i + ent_len] == ent_tokens:
                    # Check for overlap
                    if all(tags[j] == "O" for j in range(i, i + ent_len)):
                        tags[i] = f"B-{ent_label}"
                        for j in range(1, ent_len):
                            tags[i + j] = f"I-{ent_label}"

                        # Update pointer to prevent backward matching
                        search_start_idx = i + ent_len
                        break

        for tok, tag in zip(tokens, tags):
            bg = "#fff3cd" if tag != "O" else "transparent"
            border = "1px solid #ffc107" if tag != "O" else "1px solid transparent"
            bio_html += f"<span class='bio-tag' style='background:{bg}; border:{border};'><b>{tok}</b> <small style='color:#dc3545'>[{tag}]</small></span>"
        st.markdown(bio_html, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            # Assignment 1.3: Dependency Tree
            st.write("**[Tác vụ 1.3] Cây phụ thuộc (Dependency Relations)**")
            if item["dependencies"]:
                df_dep = pd.DataFrame(item["dependencies"])
                st.dataframe(
                    df_dep[["token", "relation", "head_token"]],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.write("Không có dữ liệu dependency.")

        with col_right:
            # Assignment 2.2: SRL
            st.write("**[Tác vụ 2.2] Gán nhãn vai trò ngữ nghĩa (SRL)**")
            st.json(item["srl"])
        st.divider()


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Legal Intelligence")
    st.caption("Hệ thống NLP & RAG Hợp đồng Pháp lý")
    st.divider()
    st.caption("Trạng thái hệ thống: **Online** 🟢")

# --- MAIN LAYOUT ---
tabs = st.tabs(
    [
        "📁 Raw Docs",
        "📄 Processed Docs",
        "🗃️ Database Explorer",
        "📊 Semantic Engine",
        "💬 Chatbot RAG",
    ]
)
tab_raw, tab_proc, tab_db, tab_viz, tab_chat = tabs

# --- TAB: RAW DOCUMENTS ---
with tab_raw:
    st.subheader("📦 Raw Files (data/raw)")
    with st.container(border=True):
        up_file = st.file_uploader("Upload new document", type=["pdf", "docx", "txt"])
        if st.button("🚀 Process Pipeline", type="primary"):
            if up_file:
                with st.spinner("Processing..."):
                    files = {"file": (up_file.name, up_file.getvalue())}
                    requests.post(f"{API_URL}/ingest_file", files=files)
                    st.success("Ingestion pipeline triggered!")
                    st.rerun()

    raw_files = requests.get(f"{API_URL}/documents/raw").json().get("files", [])
    for rf in raw_files:
        with st.expander(f"📄 {rf}"):
            c1, c2, c3 = st.columns([2, 1, 1])
            if c1.button("🔄 Reprocess", key=f"re_{rf}"):
                requests.post(f"{API_URL}/documents/{rf}/reprocess")
                st.rerun()
            if c2.button("📝 Rename", key=f"rn_raw_{rf}"):
                st.session_state.rename_obj = {"name": rf, "dir": "data/raw"}
            if c3.button("🗑️ Delete", key=f"dl_raw_{rf}"):
                requests.delete(f"{API_URL}/documents/raw/{rf}")
                st.rerun()

# --- TAB: PROCESSED DOCUMENTS ---
with tab_proc:
    st.subheader("⚙️ Processed Text (data/processed)")
    proc_data = requests.get(f"{API_URL}/documents/processed").json().get("files", [])
    for pf_obj in proc_data:
        pf = pf_obj["filename"]  # Assign to pf so the rest of your button logic works!
        aliases = pf_obj["aliases"]
        with st.expander(f"📄 {pf}"):
            st.caption(f"**Aliases:** {aliases}")
            c1, c2, c3 = st.columns([2, 1, 1])
            if c1.button("🔍 Visualize", key=f"viz_{pf}"):
                # Fetch pre-computed state from the database endpoint
                res = requests.get(f"{API_URL}/database/state")
                all_records = res.json().get("records", [])

                # Filter records that belong exactly to this file
                file_records = [r for r in all_records if r.get("source") == pf]

                if not file_records:
                    st.warning(
                        f"No data found in DB for {pf}. Please 'Process Pipeline' first."
                    )
                else:
                    # Reconstruct the UI data format from DB metadata

                    viz_data = []
                    for r in file_records:
                        try:
                            # Safely evaluate stringified DB lists back to Python objects
                            entities = ast.literal_eval(r.get("entities", "[]"))
                            deps = ast.literal_eval(r.get("dependencies", "[]"))
                            srl_roles = ast.literal_eval(r.get("srl_roles", "{}"))

                            # Ensure we don't double-process or lose labels during reconstruction
                            viz_data.append(
                                {
                                    "clause": r.get("document", ""),
                                    "intent": r.get("intent", ""),
                                    "entities": entities
                                    if isinstance(entities, list)
                                    else [],
                                    "np_chunks": ast.literal_eval(
                                        r.get("np_chunks", "[]")
                                    ),
                                    "dependencies": [
                                        d
                                        if isinstance(d, dict)
                                        else {
                                            "token": d.split("(")[0]
                                            if isinstance(d, str) and "(" in d
                                            else str(d),
                                            "relation": d.split("(")[1][:-1]
                                            if isinstance(d, str) and "(" in d
                                            else "",
                                            "head_token": "",
                                        }
                                        for d in deps
                                    ],
                                    "srl": {
                                        "predicate": r.get("predicate", ""),
                                        "roles": srl_roles,
                                    },
                                }
                            )
                        except Exception:
                            pass

                    st.session_state.viz_data = viz_data
                    st.session_state.viz_name = pf
                    st.success("Data loaded from DB! Ready in Semantic Engine tab")
            if c2.button("📝 Rename", key=f"rn_proc_{pf}"):
                st.session_state.rename_obj = {"name": pf, "dir": "data/processed"}
            if c3.button("🗑️ Delete", key=f"dl_proc_{pf}"):
                requests.delete(f"{API_URL}/documents/processed/{pf}")
                st.rerun()

if "rename_obj" in st.session_state:
    with st.sidebar.form("rename_form_global"):
        obj = st.session_state.rename_obj
        st.write(f"Renaming in **{obj['dir']}**")
        new_name = st.text_input("New filename:", value=obj["name"])
        if st.form_submit_button("Confirm"):
            requests.post(
                f"{API_URL}/documents/rename?target_dir={obj['dir']}&old_name={obj['name']}&new_name={new_name}"
            )
            del st.session_state.rename_obj
            st.rerun()

# --- TAB: DATABASE EXPLORER ---
with tab_db:
    st.header("Vector DB Records")

    if st.button("🔄 Reload Database"):
        res = requests.get(f"{API_URL}/database/state").json()
        st.session_state.full_df = pd.DataFrame(res.get("records", []))
        st.rerun()

    if "full_df" not in st.session_state:
        res = requests.get(f"{API_URL}/database/state").json()
        st.session_state.full_df = pd.DataFrame(res.get("records", []))

    df = st.session_state.full_df
    if df.empty:
        st.info("Database is empty")
    else:
        # Optimization: Filter and search
        search_q = st.text_input("Search in clauses:")
        filtered_df = (
            df[df["document"].str.contains(search_q, case=False)] if search_q else df
        )

        # Super-fast pagination using Streamlit native dataframe features
        st.dataframe(filtered_df, width="stretch", height=600, hide_index=True)

        # Source-based deletion
        st.divider()
        st.subheader("Manage Vector Sources")
        sources_in_db = (
            requests.get(f"{API_URL}/database/sources").json().get("sources", [])
        )
        for sdb in sources_in_db:
            c1, c2 = st.columns([4, 1])
            c1.write(f"Source: `{sdb}`")
            if c2.button("🗑️ Wipe Vectors", key=f"wipe_{sdb}"):
                requests.delete(f"{API_URL}/database/source/{sdb}")
                st.rerun()

# --- TAB 3: SEMANTIC ENGINE (VISUALIZER) ---
with tab_viz:
    if "viz_data" in st.session_state:
        st.header(f"Chi tiết Phân tích: {st.session_state.viz_name}")
        render_nlp_dashboard(st.session_state.viz_data)
    else:
        st.info(
            "Chưa có dữ liệu để hiển thị. Hãy vào tab 'Quản lý Hợp đồng' và bấm nút 'Visualize' ở một hợp đồng bất kỳ."
        )

# --- TAB 4: RAG Q&A ---
with tab_chat:
    st.header("Hỏi Đáp Hợp Đồng Thông Minh (RAG)")
    st.markdown(
        "Hệ thống sẽ truy xuất các mệnh đề chính xác nhất trong DB để trả lời câu hỏi của bạn."
    )

    sources = fetch_sources()
    # available_filters now correctly handles the list of strings returned by the API
    available_filters = ["Tất cả Hợp đồng"] + sources

    colA, colB = st.columns([3, 1])
    with colA:
        question = st.text_input(
            "Đặt câu hỏi:",
            placeholder="VD: Điều kiện chấm dứt hợp đồng thuê nhà là gì?",
        )
    with colB:
        selected_source = st.selectbox("Phạm vi tìm kiếm:", available_filters)

    # Map back to API format
    api_source_filter = (
        None if selected_source == "Tất cả Hợp đồng" else selected_source
    )

    if st.button("Truy vấn Hệ thống", type="primary"):
        if not question:
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang tìm kiếm trong DB và Gọi Gemini..."):
                try:
                    payload = {"question": question, "source_filter": api_source_filter}
                    res = requests.post(f"{API_URL}/ask", json=payload).json()

                    st.markdown(f"### 🤖 Trả lời\n{res['answer']}")

                    c1, c2 = st.columns(2)
                    with c1:
                        with st.expander("🔍 Phase 1: Routing Debug (Document Picker)"):
                            st.code(res.get("routing_debug"), language="markdown")
                    with c2:
                        with st.expander("🛠️ Phase 2: RAG Prompt (Semantic Context)"):
                            st.code(res.get("debug_prompt"), language="markdown")

                    st.markdown("### 📚 Nguồn trích dẫn (Citations)")
                    for i, src in enumerate(res["sources"]):
                        meta = src["metadata"]

                        # Extract retrieval scores mapped from backend
                        score_total = meta.get("score_total", "N/A")
                        score_vector = meta.get("score_vector", "N/A")
                        score_srl = meta.get("score_srl", "N/A")

                        # Create HTML badges for scores
                        score_html = f"""
                        <div style='margin: 8px 0; font-size: 0.85em; color: #495057;'>
                            <span style='background: #e9ecef; padding: 3px 8px; border-radius: 12px; margin-right: 5px; border: 1px solid #ced4da;'>🏆 Total: <b>{score_total}</b></span>
                            <span style='background: #e2e3e5; padding: 3px 8px; border-radius: 12px; margin-right: 5px;'>🤖 Vector: {score_vector}</span>
                            <span style='background: #e2e3e5; padding: 3px 8px; border-radius: 12px;'>🧠 SRL Heuristic: {score_srl}</span>
                        </div>
                        """

                        # Clean markdown-based citation to avoid tag leakage
                        st.markdown(
                            f"**[{i + 1}] File:** `{meta.get('source')}` | **Vị trí:** `{meta.get('context', 'Chung')}`"
                        )

                        # Display semantic matching breakdown for transparency
                        srl_raw_meta = meta.get("score_srl_breakdown", "{}")
                        try:
                            # Use safe evaluation for internal metadata string
                            breakdown = (
                                ast.literal_eval(srl_raw_meta)
                                if isinstance(srl_raw_meta, str)
                                else srl_raw_meta
                            )

                            # Bỏ điều kiện predicate_match > 0 để hiện toàn bộ breakdown
                            if (
                                isinstance(breakdown, dict)
                                and "predicate_match" in breakdown
                            ):
                                with st.expander("📊 Semantic Analysis Details"):
                                    st.write(
                                        f"**Predicate Similarity:** {breakdown.get('predicate_match', 0)}"
                                    )
                                    if breakdown.get("role_matches"):
                                        st.write("**Role Alignments:**")
                                        st.json(breakdown.get("role_matches"))
                                    st.caption(
                                        f"Final Semantic Weight: {breakdown.get('role_final_score', 0)}"
                                    )
                        except Exception:
                            st.caption("Detailed scores unavailable for this record.")

                        st.write(
                            f"> {src['content']}"
                        )  # Use standard markdown blockquote
                        st.markdown(score_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Lỗi kết nối API: {e}")
