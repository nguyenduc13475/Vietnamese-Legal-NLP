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
    .metric-container {background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;}
    </style>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR: GLOBAL INGESTION & SETTINGS ---
with st.sidebar:
    st.title("⚖️ Legal NLP Workspace")
    st.markdown("Enterprise Contract Intelligence")
    st.divider()

    st.header("📥 Document Ingestion")
    st.caption(
        "Upload contracts to index them into the Vector Database for semantic search."
    )
    uploaded_file = st.file_uploader(
        "Upload Contract", type=["txt", "pdf", "docx"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("Index Document", type="primary", use_container_width=True):
            with st.spinner("Processing & Indexing..."):
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
                        st.success(f"Indexed {res.json()['num_clauses']} clauses!")
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except Exception:
                    st.error("API Connection Error. Ensure Backend is running.")

    st.divider()
    st.caption("System Status: **Online** 🟢")

# --- MAIN LAYOUT ---
tab1, tab2 = st.tabs(["📊 Semantic Dashboard", "💬 Intelligent Q&A"])

# --- TAB 1: SEMANTIC DASHBOARD ---
with tab1:
    st.header("Real-time Contract Analysis")
    st.markdown(
        "Extract structured metadata, intents, and obligations from raw contract text."
    )

    default_text = "Bên B sẽ thanh toán toàn bộ tiền thuê 10,000,000 VNĐ trước ngày 5 hàng tháng, và nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng."
    contract_text = st.text_area(
        "Input Contract Draft:", height=120, value=default_text
    )

    if st.button("Run Semantic Extraction", type="primary"):
        with st.spinner("Analyzing semantics via PhoBERT..."):
            try:
                res = requests.post(f"{API_URL}/extract", json={"text": contract_text})
                if res.status_code == 200:
                    data = res.json().get("results", [])

                    if not data:
                        st.warning("No clauses detected.")
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

                        # -- VISUALIZATIONS --
                        vcol1, vcol2 = st.columns(2)

                        with vcol1:
                            st.subheader("Clause Intent Distribution")
                            intent_df = (
                                pd.DataFrame({"Intent": intents})
                                .value_counts()
                                .reset_index()
                            )
                            intent_df.columns = ["Intent", "Count"]
                            st.bar_chart(intent_df.set_index("Intent"), height=250)

                        with vcol2:
                            st.subheader("Entity Recognition Density")
                            all_entities = []
                            for item in data:
                                all_entities.extend(
                                    [e["label"] for e in item["entities"]]
                                )

                            if all_entities:
                                ent_df = (
                                    pd.DataFrame({"Entity Type": all_entities})
                                    .value_counts()
                                    .reset_index()
                                )
                                ent_df.columns = ["Entity Type", "Count"]
                                st.bar_chart(
                                    ent_df.set_index("Entity Type"), height=250
                                )
                            else:
                                st.info("No entities to chart.")

                        # -- DETAILED DATA TABLE --
                        st.subheader("Extraction Ledger")
                        df = pd.DataFrame(
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
                        st.dataframe(df, use_container_width=True, hide_index=True)

                else:
                    st.error(f"Extraction failed: {res.text}")
            except Exception:
                st.error("API Connection Error. Ensure FastAPI backend is running.")

# --- TAB 2: RAG Q&A ---
with tab2:
    st.header("Context-Aware Contract Q&A")
    st.markdown("Query your indexed repository using natural language.")

    # Fetch available sources dynamically
    available_sources = ["All Contracts"]
    try:
        src_res = requests.get(f"{API_URL}/sources")
        if src_res.status_code == 200:
            fetched = src_res.json().get("sources", [])
            if fetched:
                available_sources.extend(fetched)
    except Exception:
        pass  # Silently fail and just show "All Contracts" if API is down

    colA, colB = st.columns([3, 1])
    with colA:
        question = st.text_input(
            "Ask a question about the contracts:",
            placeholder="e.g., Điều kiện chấm dứt hợp đồng là gì?",
        )
    with colB:
        selected_source = st.selectbox("Target Document:", available_sources)

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
                            "🔍 View Retrieved Sources (Vector DB)", expanded=False
                        ):
                            for idx, src in enumerate(ans_data["sources"]):
                                st.markdown(f"**[{idx + 1}]** {src}")
                    else:
                        st.error(f"Query failed: {res.text}")
                except Exception:
                    st.error("API Connection Error. Ensure FastAPI backend is running.")
