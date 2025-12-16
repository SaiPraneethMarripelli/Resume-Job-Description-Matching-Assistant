# Imports
import os
import asyncio
import tempfile
import streamlit as st
# Fix event loop (Gemini + Streamlit)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# CONFIG
os.environ["GOOGLE_API_KEY"] = "AIzaSyCx8-uLdL2lQBCTd-bBfOXcNoGjZy1lsm0"
# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
# HELPER: Save uploaded file
def save_uploaded_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
# STREAMLIT UI
st.title("📄 Resume-Job Description Matching Assistant")

st.subheader("Upload Resume and Job Description")

resume_file = st.file_uploader(
    "Upload Resume (PDF or TXT)",
    type=["pdf", "txt"]
)

jd_file = st.file_uploader(
    "Upload Job Description (PDF or TXT)",
    type=["pdf", "txt"]
)
# ANALYSIS
if st.button("Analyze Resume Match"):

    if resume_file is None or jd_file is None:
        st.error("❌ Please upload both Resume and Job Description.")
        st.stop()

    # Save uploaded files
    resume_path = save_uploaded_file(resume_file)
    jd_path = save_uploaded_file(jd_file)
    # Load Resume
    if resume_path.endswith(".pdf"):
        resume_docs = PyPDFLoader(resume_path).load()
    else:
        resume_docs = TextLoader(resume_path, encoding="utf-8").load()
    # Load Job Description
    if jd_path.endswith(".pdf"):
        jd_docs = PyPDFLoader(jd_path).load()
    else:
        jd_docs = TextLoader(jd_path, encoding="utf-8").load()

    jd_text = "\n".join([doc.page_content for doc in jd_docs])
    # Split Resume into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    resume_chunks = splitter.split_documents(resume_docs)
    # Embeddings + Vector Store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        resume_chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.get_relevant_documents(jd_text)

    context = "\n\n".join([doc.page_content for doc in context_docs])
    # Prompt
    prompt = f"""
You are an AI assistant that analyzes resume suitability.
Do not hallucinate. Base answers strictly on retrieved resume content.

Job Description:
{jd_text}

Relevant Resume Content:
{context}

Provide:
1. Matching skills
2. Missing skills
3. Improvement suggestions

Respond clearly in bullet points.
"""

    response = llm.invoke(prompt)

    st.subheader("📊 Resume Match Analysis")
    st.markdown(response.content)
