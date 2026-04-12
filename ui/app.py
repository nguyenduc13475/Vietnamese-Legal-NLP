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
    .status-badge-indexed {background-color: #d1e7dd; color: #0f5132; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold;}
    .status-badge-pending {background-color: #fff3cd; color: #664d03; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- UTILS ---
@st.cache_data()  # Cache persists until explicitly cleared during document ingestion to prevent random UI stutters
def fetch_sources():
    try:
        res = requests.get(f"{API_URL}/sources")
        if res.status_code == 200:
            return res.json().get("sources", [])
    except Exception:
        pass
    return []


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Legal Intelligence")
    st.caption("Enterprise NLP & RAG Platform")
    st.divider()

    st.header("📥 Ingest Document")
    st.caption(
        "Upload contracts to clean, process, and index them into the semantic database."
    )
    uploaded_file = st.file_uploader(
        "Upload Contract", type=["txt", "pdf", "docx"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("Process & Index", type="primary", width="stretch"):
            with st.spinner("Processing via LLM & Indexing..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    res = requests.post(f"{API_URL}/ingest_file", files=files)
                    if res.status_code == 200:
                        st.toast(
                            f"Indexed {res.json()['num_clauses']} clauses successfully!",
                            icon="✅",
                        )
                        st.cache_data.clear()  # Force refresh source list
                        if "db_data" in st.session_state:
                            del st.session_state[
                                "db_data"
                            ]  # Clear cached DB records so it re-fetches
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except Exception:
                    st.error("API Connection Error. Ensure Backend is running.")

    st.divider()
    st.caption("System Status: **Online** 🟢")

# --- MAIN LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗄️ Document Hub", "🗃️ DB Explorer", "📊 Semantic Engine", "💬 Intelligent Q&A"]
)

# --- TAB 1: DOCUMENT HUB ---
with tab1:
    st.header("Document Repository")
    st.markdown(
        "Manage your processed legal contracts and monitor vector indexing status."
    )

    sources = fetch_sources()

    if not sources:
        st.info(
            "No processed documents found. Upload a document via the sidebar to begin."
        )
    else:
        # Format data for an enterprise table
        table_data = []
        for src in sources:
            status_html = (
                '<span class="status-badge-indexed">Indexed</span>'
                if src["indexed"]
                else '<span class="status-badge-pending">Unindexed</span>'
            )
            if src.get("missing_file"):
                status_html = '<span class="status-badge-pending" style="background-color:#f8d7da; color:#842029;">Missing Source File</span>'

            table_data.append(
                {
                    "Document Name": src["filename"],
                    "Vector DB Status": status_html,
                    "Action": "Ready for Q&A" if src["indexed"] else "Needs Indexing",
                }
            )

        df_docs = pd.DataFrame(table_data)
        st.write(df_docs.to_html(escape=False, index=False), unsafe_allow_html=True)

# --- TAB 2: DB EXPLORER ---
with tab2:
    st.header("Database Explorer")
    st.markdown(
        "Inspect the raw vectorized clauses and metadata currently residing in ChromaDB."
    )

    if "db_page" not in st.session_state:
        st.session_state.db_page = 1

    def reset_page():
        st.session_state.db_page = 1

    def fetch_all_db_records():
        try:
            res = requests.get(f"{API_URL}/database/state")
            if res.status_code == 200:
                return res.json()
        except Exception:
            return None
        return None

    # Fetch once and store in session state to avoid lag on page change
    if "db_data" not in st.session_state:
        with st.spinner("Fetching all records from database..."):
            st.session_state.db_data = fetch_all_db_records()

    # Control Bar
    col1, col2, col3 = st.columns([2, 5, 2])
    with col1:
        page_size_val = st.selectbox(
            "Rows per page", [100, 500, 1000, "All"], index=0, on_change=reset_page
        )
    with col3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", width="stretch"):
            with st.spinner("Refreshing database records..."):
                st.session_state.db_data = fetch_all_db_records()
            st.cache_data.clear()
            st.rerun()

    db_data = st.session_state.db_data

    if not db_data or (not db_data.get("records") and db_data.get("total", 0) == 0):
        st.info("The Vector Database is currently empty or unreachable.")
    else:
        records = db_data.get("records", [])
        total_records = len(records)
        limit = total_records if page_size_val == "All" else int(page_size_val)

        if limit <= 0:
            limit = 1

        max_page = max(1, (total_records + limit - 1) // limit)

        # Pagination Callbacks
        def go_first():
            st.session_state.db_page = 1

        def go_prev():
            st.session_state.db_page -= 1

        def go_next():
            st.session_state.db_page += 1

        def go_last():
            st.session_state.db_page = max_page

        def set_page():
            st.session_state.db_page = st.session_state.page_input

        # Pagination Controls
        col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(
            [1, 1, 1.5, 1, 1, 2.5]
        )

        with col_p1:
            st.write("")
            st.button(
                "⏮ First",
                disabled=(st.session_state.db_page <= 1),
                width="stretch",
                on_click=go_first,
            )
        with col_p2:
            st.write("")
            st.button(
                "◀ Prev",
                disabled=(st.session_state.db_page <= 1),
                width="stretch",
                on_click=go_prev,
            )
        with col_p3:
            st.number_input(
                "Page:",
                min_value=1,
                max_value=max_page,
                value=min(st.session_state.db_page, max_page),
                key="page_input",
                on_change=set_page,
            )
        with col_p4:
            st.write("")
            st.button(
                "Next ▶",
                disabled=(st.session_state.db_page >= max_page),
                width="stretch",
                on_click=go_next,
            )
        with col_p5:
            st.write("")
            st.button(
                "Last ⏭",
                disabled=(st.session_state.db_page >= max_page),
                width="stretch",
                on_click=go_last,
            )
        with col_p6:
            st.write("")
            st.caption(
                f"Showing page **{min(st.session_state.db_page, max_page)}** of **{max_page}** (Total: **{total_records}** records)"
            )

        # Slice Data Instantly (in-memory)
        current_page = min(st.session_state.db_page, max_page)
        offset = (current_page - 1) * limit
        page_records = records[offset : offset + limit]

        df_db = pd.DataFrame(page_records)
        if not df_db.empty:
            df_db = df_db[
                [
                    "source",
                    "intent",
                    "document",
                    "entities",
                    "predicate",
                    "srl_roles",
                    "dependencies",
                    "id",
                ]
            ]

            # Ensure empty dicts/lists or missing values are clearly displayed as N/A
            for col in ["entities", "srl_roles", "dependencies"]:
                df_db[col] = (
                    df_db[col].astype(str).replace(["{}", "[]", "", "None"], "N/A")
                )

            df_db.columns = [
                "Source File",
                "Intent",
                "Clause Text",
                "Entities",
                "Predicate",
                "SRL Roles",
                "Dependencies",
                "Chroma ID",
            ]

            st.dataframe(
                df_db,
                width="stretch",
                hide_index=True,
                height=600,
                column_config={
                    "Clause Text": st.column_config.TextColumn(width="large"),
                    "Entities": st.column_config.TextColumn(width="large"),
                    "SRL Roles": st.column_config.TextColumn(width="large"),
                    "Dependencies": st.column_config.TextColumn(width="large"),
                },
            )

# --- TAB 3: SEMANTIC ENGINE ---
with tab3:
    st.header("Real-time NLP Extraction")
    st.markdown(
        "Extract structured metadata, intents, and obligations instantly on CPU."
    )

    default_text = "Bên B sẽ thanh toán toàn bộ tiền thuê 10,000,000 VNĐ trước ngày 5 hàng tháng, và nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng."
    contract_text = st.text_area(
        "Input Contract Clause(s):", height=120, value=default_text
    )

    if st.button("Run Extraction", type="primary"):
        with st.spinner("Analyzing semantics..."):
            try:
                res = requests.post(f"{API_URL}/extract", json={"text": contract_text})
                if res.status_code == 200:
                    data = res.json().get("results", [])

                    if not data:
                        st.warning("No valid legal clauses detected.")
                    else:
                        st.success("Extraction Complete.")

                        # -- TOP LEVEL METRICS --
                        col1, col2, col3, col4 = st.columns(4)
                        total_clauses = len(data)
                        total_entities = sum(len(item["entities"]) for item in data)
                        intents = [item["intent"] for item in data]
                        penalties = sum(
                            1
                            for item in data
                            if any(e["label"] == "PENALTY" for e in item["entities"])
                        )

                        col1.metric("Total Clauses", total_clauses)
                        col2.metric("Entities Extracted", total_entities)
                        col3.metric("Obligations Found", intents.count("Obligation"))
                        col4.metric("Penalty Clauses", penalties)

                        st.divider()

                        # -- DETAILED DATA TABLE --
                        st.subheader("Extraction Ledger")
                        df_extract = pd.DataFrame(
                            [
                                {
                                    "Clause": item["clause"],
                                    "Intent": item["intent"],
                                    "Entities": ", ".join(
                                        set(e["label"] for e in item["entities"])
                                    ),
                                    "Predicate (SRL)": item["srl"].get("predicate", ""),
                                }
                                for item in data
                            ]
                        )
                        st.dataframe(df_extract, width="stretch", hide_index=True)
                else:
                    st.error(f"Extraction failed: {res.text}")
            except Exception:
                st.error("API Connection Error. Ensure FastAPI backend is running.")

# --- TAB 4: RAG Q&A ---
with tab4:
    st.header("Context-Aware Q&A")
    st.markdown("Query your indexed legal repository using natural language.")

    sources = fetch_sources()
    # Only allow filtering by documents that are actually indexed
    available_filters = ["All Contracts"] + [
        s["filename"] for s in sources if s.get("indexed")
    ]

    colA, colB = st.columns([3, 1])
    with colA:
        question = st.text_input(
            "Ask a question about the contracts:",
            placeholder="e.g., Điều kiện chấm dứt hợp đồng là gì?",
        )
    with colB:
        selected_source = st.selectbox("Target Document:", available_filters)

    if st.button("Query Database", type="primary"):
        if not question:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching Vector Database & Generating Response..."):
                try:
                    payload = {"question": question, "source_filter": selected_source}
                    res = requests.post(f"{API_URL}/ask", json=payload)

                    if res.status_code == 200:
                        ans_data = res.json()
                        st.info(f"**AI Response:**\n\n{ans_data['answer']}")

                        with st.expander(
                            "🔍 View Retrieved Sources & Citations", expanded=False
                        ):
                            for idx, src in enumerate(ans_data["sources"]):
                                st.markdown(f"**[{idx + 1}]** {src}")
                                st.divider()
                    else:
                        st.error(f"Query failed: {res.text}")
                except Exception:
                    st.error("API Connection Error. Ensure FastAPI backend is running.")
