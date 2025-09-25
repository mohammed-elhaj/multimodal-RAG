#!/usr/bin/env python3
"""
Main Streamlit application for the Ashraq RAG Agent - Simplified Version.
"""
import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    PAGE_TITLE, PAGE_ICON, COLLECTION_NAME,
    CHROMA_DB_PATH
)

# --- 1. Page and System Configuration ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title(f"{PAGE_ICON} Ashraq RAG Agent")
load_dotenv()

# --- PDF to Image Conversion Function ---
@st.cache_resource
def convert_pdf_to_images():
    """
    Convert PDF to PNG images on startup if images don't exist.
    This function runs once and is cached.
    """
    pdf_path = Path("./data/task-mohamed-rag.pdf")
    images_dir = Path("./images")
    
    # Check if images directory exists and has PNG files
    if images_dir.exists():
        existing_images = list(images_dir.glob("page_*.png"))
        if existing_images:
            st.info(f"✅ Found {len(existing_images)} existing page images. Skipping conversion.")
            return
    
    # Create images directory if it doesn't exist
    images_dir.mkdir(exist_ok=True)
    
    # Check if PDF exists
    if not pdf_path.exists():
        st.warning(f"⚠️ PDF file not found at {pdf_path}")
        return
    
    try:
        # Open PDF document
        pdf_document = fitz.open(str(pdf_path))
        total_pages = len(pdf_document)
        
        st.info(f"🔄 Converting {total_pages} PDF pages to images...")
        
        # Convert each page to image
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Render page to pixmap (image)
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image and save
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save as page_1.png, page_2.png, etc. (1-indexed)
            image_path = images_dir / f"page_{page_num + 1}.png"
            img.save(image_path, "PNG")
        
        pdf_document.close()
        st.success(f"✅ Successfully converted {total_pages} pages to PNG images!")
        
    except Exception as e:
        st.error(f"❌ Error converting PDF to images: {e}")

# Run PDF to image conversion on startup
convert_pdf_to_images()

# --- 2. RAG Chain Setup ---
@st.cache_resource
def get_rag_components():
    """
    Creates and caches the RAG components (vector store and LLM).
    Returns the vector store and LLM for dynamic retrieval and chain creation.
    """
    # Initialize LLM and embedding models
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Load the persisted vector store
    vector_store = Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME
    )
    
    return vector_store, llm

# --- 3. Dynamic Chunk Retrieval Function ---
def retrieve_dynamic_chunks(vector_store, query, min_k=3, max_k=12, relevance_threshold=0.4):
    """
    Dynamically retrieves chunks based on relevance scores.
    
    Args:
        vector_store: The Chroma vector store
        query: User query string
        min_k: Minimum number of chunks to retrieve (default: 3)
        max_k: Maximum number of chunks to retrieve (default: 12)
        relevance_threshold: Minimum relevance score to include a chunk (default: 0.4)
    
    Returns:
        List of relevant documents with scores
    """
    # Retrieve more chunks initially to evaluate relevance
    results_with_scores = vector_store.similarity_search_with_relevance_scores(
        query,
        k=max_k
    )
    
    # Filter based on relevance threshold and ensure minimum chunks
    filtered_results = []
    for doc, score in results_with_scores:
        if score >= relevance_threshold or len(filtered_results) < min_k:
            filtered_results.append((doc, score))
    
    # Calculate relevance distribution
    high_relevance_count = sum(1 for _, score in filtered_results if score >= 0.65)
    medium_relevance_count = sum(1 for _, score in filtered_results if 0.5 <= score < 0.65)
    
    # Adaptive logic: Include more chunks based on relevance distribution
    if high_relevance_count >= 2:
        # If we have multiple high-relevance chunks, be more inclusive
        adaptive_threshold = relevance_threshold * 0.85
        filtered_results = [
            (doc, score) for doc, score in results_with_scores
            if score >= adaptive_threshold
        ][:max_k]
    elif medium_relevance_count >= 3:
        # If we have several medium-relevance chunks, include more of them
        adaptive_threshold = relevance_threshold * 0.9
        filtered_results = [
            (doc, score) for doc, score in results_with_scores
            if score >= adaptive_threshold
        ][:max(8, min(max_k, len(results_with_scores)))]
    
    # Ensure we always return at least min_k results
    if len(filtered_results) < min_k:
        filtered_results = results_with_scores[:min_k]
    
    return filtered_results

# --- 4. Parse Citations from Answer ---
def extract_cited_pages(answer_text):
    """
    Extract page numbers that are actually cited in the answer text.
    Looks for patterns like [Page X], (Page X), "Page X", and Sources: Page X, Y, Z
    
    Args:
        answer_text: The generated answer text
    
    Returns:
        List of unique page numbers cited in the answer
    """
    import re
    
    cited_pages = set()
    
    # Pattern 1: [Page X] or (Page X)
    pattern1 = r'[\[\(]Page\s+(\d+)[\]\)]'
    matches1 = re.findall(pattern1, answer_text, re.IGNORECASE)
    cited_pages.update(int(m) for m in matches1)
    
    # Pattern 2: Sources: Page X, Y, Z
    pattern2 = r'Sources?:\s*Page[s]?\s*([\d,\s]+)'
    matches2 = re.findall(pattern2, answer_text, re.IGNORECASE)
    for match in matches2:
        # Extract individual page numbers from comma-separated list
        page_nums = re.findall(r'\d+', match)
        cited_pages.update(int(p) for p in page_nums)
    
    # Pattern 3: "Page X" in text
    pattern3 = r'\bPage\s+(\d+)\b'
    matches3 = re.findall(pattern3, answer_text, re.IGNORECASE)
    cited_pages.update(int(m) for m in matches3)
    
    return sorted(list(cited_pages))

# --- 5. Enhanced RAG Workflow Function ---
def process_query(user_query):
    """
    Processes a user query through the enhanced RAG workflow:
    1. Dynamically retrieve relevant chunks based on relevance scores
    2. Generate text answer using LLM
    3. Extract actually cited pages from the answer
    4. Return answer and only the cited page numbers for image display
    """
    try:
        # Get the RAG components
        vector_store, llm = get_rag_components()
        
        # Step 1: Dynamically retrieve relevant document chunks
        results_with_scores = retrieve_dynamic_chunks(
            vector_store,
            user_query,
            min_k=3,  # Minimum 3 chunks (matches original behavior)
            max_k=12,  # Maximum 12 chunks (allows for more comprehensive context)
            relevance_threshold=0.4  # Lower threshold to be more inclusive
        )
        
        # Extract just the documents for the chain
        retrieved_docs = [doc for doc, score in results_with_scores]
        
        # Log retrieval info for debugging
        st.sidebar.info(f"Retrieved {len(retrieved_docs)} chunks based on relevance")
        
        # Format documents with page numbers
        def format_docs_with_page_numbers(docs):
            """Format retrieved documents to include page numbers."""
            formatted = []
            for doc in docs:
                page_num = doc.metadata.get('page_number', 'Unknown')
                content = doc.page_content
                formatted.append(f"[Page {page_num}]: {content}")
            return "\n\n".join(formatted)
        
        # Create the prompt template
        template = """You are an expert on Ashraq's communication strategy. Answer the user's question based *only* on the following context.
        Each piece of context is prefixed with its page number [Page X].
        
        If the context does not contain the answer, state that the information is not available in the document.
        
        IMPORTANT: At the end of your answer, you MUST cite all the page numbers you used in the format: (Sources: Page X, Y, Z)

        CONTEXT:
        {context}

        QUESTION:
        {question}
        
        Remember to include source citations at the end of your answer!
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create and run the RAG chain
        formatted_context = format_docs_with_page_numbers(retrieved_docs)
        
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        
        # Step 2: Generate the answer
        answer = chain.invoke({
            "context": formatted_context,
            "question": user_query
        })
        
        # Step 3: Extract only the pages actually cited in the answer
        cited_page_numbers = extract_cited_pages(answer)
        
        # If no pages were explicitly cited, fall back to pages from high-relevance chunks
        if not cited_page_numbers and retrieved_docs:
            # Use pages from chunks with score > 0.6
            for doc, score in results_with_scores[:3]:  # Limit to top 3
                if score > 0.6:
                    page_num = doc.metadata.get('page_number')
                    if page_num and page_num not in cited_page_numbers:
                        cited_page_numbers.append(page_num)
            cited_page_numbers.sort()
        
        # Display relevance scores in sidebar for transparency
        if results_with_scores:
            with st.sidebar.expander("Chunk Relevance Scores"):
                for i, (doc, score) in enumerate(results_with_scores, 1):
                    page_num = doc.metadata.get('page_number', 'Unknown')
                    st.write(f"Chunk {i} (Page {page_num}): {score:.3f}")
        
        return answer, cited_page_numbers, retrieved_docs
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None, [], []

# --- 4. Streamlit UI and Interaction Logic ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this is an assistant message with source pages, display the expander
        if message["role"] == "assistant" and "page_numbers" in message:
            if message["page_numbers"]:
                with st.expander("View Source Pages"):
                    for page_num in message["page_numbers"]:
                        image_path = f"./images/page_{page_num}.png"
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Page {page_num}", use_column_width=True)

# React to user input
if prompt := st.chat_input("Ask about Ashraq's strategy..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the query through the simplified RAG workflow
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            # Execute the simplified workflow: Retrieve → Answer → Show Sources
            answer, page_numbers, retrieved_docs = process_query(prompt)
            
            if answer:
                # Display the text answer
                st.markdown(answer)
                
                # Always show source pages in an expander (if pages were retrieved)
                if page_numbers:
                    with st.expander("View Source Pages"):
                        for page_num in page_numbers:
                            image_path = f"./images/page_{page_num}.png"
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f"Page {page_num}", use_column_width=True)
                            else:
                                st.warning(f"Image for page {page_num} not found.")
                
                # Add assistant response to chat history with page numbers
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "page_numbers": page_numbers
                })
            else:
                error_msg = "I couldn't process your query. Please try again."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})