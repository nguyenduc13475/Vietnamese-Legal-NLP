import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Contract Analysis", layout="wide", page_icon="⚖️")

st.title("Contract Analysis & QA System")
st.markdown("Legal Contract Semantic Mining")

tab1, tab2 = st.tabs(["1. Semantic Analysis", "2. RAG Question & Answering"])

with tab1:
    st.header("Structured Information Extraction")
    default_text = "Bên B sẽ thanh toán toàn bộ tiền thuê 10,000,000 VNĐ trước ngày 5 hàng tháng, và nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng."
    contract_text = st.text_area("Enter contract text:", height=150, value=default_text)

    if st.button("Start Analysis", type="primary"):
        with st.spinner("Running NLP Pipeline..."):
            try:
                res = requests.post(f"{API_URL}/extract", json={"text": contract_text})
                if res.status_code == 200:
                    data = res.json()["results"]
                    st.success(f"Successfully analyzed {len(data)} clauses!")

                    # 1. Overview Table (Dataframe for displaying 1000+ lines without lag)
                    st.subheader("📋 Bảng Tổng quan")
                    if len(data) > 0:
                        df = pd.DataFrame(
                            [
                                {
                                    "STT": i + 1,
                                    "Clause": item["clause"],
                                    "Intent": item["intent"],
                                    "No. Entities": len(item["entities"]),
                                }
                                for i, item in enumerate(data)
                            ]
                        )

                        st.dataframe(
                            df, use_container_width=True, hide_index=True, height=250
                        )

                        csv_data = df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="Export Overview Table to CSV",
                            data=csv_data,
                            file_name="legal_contract_analysis.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning(
                            "No clauses could be extracted from the provided data."
                        )

                    # 2. Selective detail view interface
                    st.subheader("Clause Details")
                    if len(data) > 0:
                        selected_idx = st.selectbox(
                            "Select a clause to view detailed analysis:",
                            range(len(data)),
                            format_func=lambda x: (
                                f"Clause {x + 1}: {data[x]['clause'][:80]}..."
                            ),
                        )

                        item = data[selected_idx]
                        st.markdown(f"**Content:** {item['clause']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Intent:**")
                            st.info(item["intent"])

                            st.markdown("**NER:**")
                            if item["entities"]:
                                st.json(item["entities"])
                            else:
                                st.write("No entities found.")

                        with col2:
                            st.markdown("**SRL:**")
                            st.json(item["srl"])

                            st.markdown("**Dependency Analysis (Top 5):**")
                            if item["dependencies"]:
                                st.write(item["dependencies"][:5])
                            else:
                                st.write("No syntactic analysis data available.")
                else:
                    st.error(f"API Error: {res.text}")
            except Exception:
                st.error(
                    "Unable to connect to the API. Please ensure you have run uvicorn api.main:app --reload --port 8000."
                )

with tab2:
    st.header("Contract Query (Chatbot)")

    uploaded_file = st.file_uploader(
        "Upload your contract (.txt, .pdf, .docx) for immediate analysis:",
        type=["txt", "pdf", "docx"],
    )
    if uploaded_file is not None:
        if st.button("Load into database"):
            with st.spinner("Processing and loading data into Vector DB..."):
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
                        st.success(
                            f"Successfully loaded {res.json()['num_clauses']} clauses from {uploaded_file.name}! You can start your Q&A now."
                        )
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception:
                    st.error(
                        "Unable to connect to the API. Please ensure that the FastAPI Backend is running."
                    )

    st.markdown("---")
    st.info(
        "Note: The AI Assistant will answer based on the available template contracts and any contracts you have just uploaded above."
    )

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: Điều kiện chấm dứt hợp đồng này là gì?",
    )

    if st.button("Query", type="primary"):
        if not question:
            st.warning("Please enter your query.")
        else:
            with st.spinner("Searching for information and generating a response..."):
                try:
                    res = requests.post(f"{API_URL}/ask", json={"question": question})
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"**Response:**\n\n{data['answer']}")

                        st.markdown("**Source Extraction (From VectorDB):**")
                        for idx, src in enumerate(data["sources"]):
                            st.caption(f"{idx + 1}. {src}")
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception:
                    st.error(
                        "Unable to connect to the API. Please ensure that the FastAPI Backend is running."
                    )
