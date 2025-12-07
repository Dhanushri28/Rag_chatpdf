import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index, retrieve_relevant_docs
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

st.set_page_config(
    page_title="Chat Pro  Document Assistant",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

# Sidebar for document upload
with st.sidebar:
    st.markdown("### Document Upload")
    st.markdown("Upload your documents to start chatting!")

uploaded_files = pdf_uploader()

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) uploaded")

# Process documents
if st.button("Process Documents", type="primary"):
    if uploaded_files:
        with st.spinner("Processing your  documents..."):
            # Extract text from all PDFs
            all_texts = [extract_text_from_pdf(file) for file in uploaded_files]

            # Split texts into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = []
            for text in all_texts:
                chunks.extend(text_splitter.split_text(text))

            # Create FAISS index
            vectorstore = create_faiss_index(chunks)
            st.session_state.vectorstore = vectorstore

            # Initialize chat model
            chat_model = get_chat_model(EURI_API_KEY)
            st.session_state.chat_model = chat_model

        st.success("Documents processed successfully!")
        st.balloons()
    else:
        st.error("Please upload at least one document before processing!")

# Main chat interface
st.markdown("### Let's Chat with Your Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    # Generate assistant response
    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                # Retrieve relevant documents
                relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)

                # Create context from relevant documents
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Create prompt with context
                system_prompt = f"""You are Chat Pro, an intelligent  document assistant.

Based on the following  documents, provide accurate and helpful answers.
If the information is not in the documents, clearly state that.

 Documents:
{context}

User Question: {prompt}

Answer:"""

                response = ask_chat_model(st.session_state.chat_model, system_prompt)

            st.markdown(response)
            st.caption(timestamp)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp
            })
    else:
        with st.chat_message("assistant"):
            st.error("Please upload and process documents first!")
            st.caption(timestamp)

# Footer
st.markdown("---")
st.markdown("Chat Pro © 2025")
