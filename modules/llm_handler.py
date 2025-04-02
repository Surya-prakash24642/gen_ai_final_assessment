import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_llm(query, retrieved_texts):
    """Generates an LLM answer using Gemini API based on retrieved documents."""
    if not retrieved_texts:
        return "No relevant information found in the selected source."

    # Prepare context
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
