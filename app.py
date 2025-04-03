import os
import fitz  # PyMuPDF
import chromadb  # Vector database
import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embedding model & ChromaDB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./vector_db")  # Persistent storage
collection = chroma_client.get_or_create_collection("data_embeddings")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "query" not in st.session_state:
    st.session_state.query = ""  
if "first_question_asked" not in st.session_state:
    st.session_state.first_question_asked = False  

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texts = [page.get_text("text") for page in doc]
        return texts  
    except Exception as e:
        return [f"Error extracting text: {e}"]

# Extract text from a website
def extract_text_from_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except requests.exceptions.RequestException as e:
        return f"Error fetching website data: {e}"

# Store embeddings in ChromaDB
def store_vectors_in_chromadb(text_chunks, source_name):
    """Stores text chunks in ChromaDB with metadata."""
    for i, chunk in enumerate(text_chunks):
        collection.add(
            ids=[f"{source_name}_chunk_{i}"],  # Unique ID for each chunk
            documents=[chunk],
            metadatas=[{"source": source_name}],
        )

# Retrieve relevant text chunks
def retrieve_relevant_chunks(query):
    """Finds relevant text chunks from both PDFs and websites."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if not results["documents"]:
        return []
    
    references = []
    for doc, metadata in zip(results["documents"], results["metadatas"]):
        source_name = metadata[0].get("source", "Unknown Source") if metadata and metadata[0] else "Unknown Source"
        references.append((source_name, doc))
    
    return references  

# Generate response using Gemini
def ask_llm(query, retrieved_texts):
    """Generates an LLM answer with references."""
    if not retrieved_texts:
        return "No relevant information found in the selected source."

    # Prepare context and references
    context = "\n".join([chunk for _, texts in retrieved_texts for chunk in texts])
    references = "\n\n".join([f"**{source}**\n{text}" for source, text in retrieved_texts])

    prompt = f"""
    Based on the following document context:
    {context}

    Answer concisely:
    {query}

    Also, include references from the documents.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return f"**Answer:**\n{response.text.strip()}"

# Main App
def main():
    # Set Streamlit page to wide mode
    st.set_page_config(page_title="PDF & Website QA", layout="wide")

    # Custom CSS for extra-wide content
    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 95%;  /* Adjust width as needed */
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📄 PDF & 🌍 Website Question-Answering App")

    # Create columns for PDF & Website input
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Use PDF Upload"):
            st.session_state["use_pdf"] = True
            st.session_state["use_website"] = False

    with col2:
        if st.button("🌍 Use Website Link"):
            st.session_state["use_website"] = True
            st.session_state["use_pdf"] = False

    # Ensure default values exist
    if "use_pdf" not in st.session_state:
        st.session_state["use_pdf"] = False
    if "use_website" not in st.session_state:
        st.session_state["use_website"] = False

    # Show PDF uploader if chosen
    if st.session_state["use_pdf"]:
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            os.makedirs("uploaded_files", exist_ok=True)  # Ensure folder exists
            for uploaded_file in uploaded_files:
                pdf_path = f"uploaded_files/{uploaded_file.name}"  
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                text_chunks = extract_text_from_pdf(pdf_path)
                store_vectors_in_chromadb(text_chunks, uploaded_file.name)  # Store embeddings for each file

            st.success("All PDFs have been vectorized and stored!")

    # Show Website input if chosen
# Add inside `main()` function where website input is handled
# Show Website input if chosen
    if st.session_state["use_website"]:
        website_url = st.text_input("Enter a website URL:")

        # Show "Process URL" button immediately
        process_clicked = st.button("Process URL")

        if process_clicked and website_url:
            st.session_state["scraped_content"] = extract_text_from_website(website_url)
            
            if "Error fetching website data" in st.session_state["scraped_content"]:
                st.error("Failed to fetch website data. Check the URL and try again.")
            else:
                store_vectors_in_chromadb([st.session_state["scraped_content"]], website_url)  # Store embeddings
                st.success("✅ Web scraping completed! Data is now stored for answering questions.")


    # Ensure scraped content is used for Q&A
    if "scraped_content" not in st.session_state:
        st.session_state["scraped_content"] = ""


    # Display chat history
    st.subheader("Chat History")
    for i, (q, a) in enumerate(st.session_state.conversation_history):
        with st.expander(f"**Q{i+1}:** {q}"):
            st.write(f"**A{i+1}:** {a}")

    # User enters a question
    query = st.text_input("🔍 Ask a question from uploaded data:", value=st.session_state.query)

    if st.button("Ask"):
        if query.strip():
            relevant_texts = retrieve_relevant_chunks(query)
            answer = ask_llm(query, relevant_texts)

            st.subheader("🤖 Answer:")
            st.write(answer)

            # Show references in dropdowns
            st.subheader("📑 References:")
            for source_name, texts in relevant_texts:
                with st.expander(f"📄 {source_name}"):
                    for chunk in texts:
                        st.write(chunk)

            # Store question-answer history
            st.session_state.conversation_history.append((query, answer))
            st.session_state.first_question_asked = True  # Set flag

        else:
            st.error("Please enter a question.")

    # "Ask New Question" button after the first question
    if st.session_state.first_question_asked:
        if st.button("Ask New Question"):
            st.session_state.query = ""  # Clear the input field
            st.session_state.first_question_asked = False  # Reset flag

if __name__ == "__main__":
    main()
