import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # Fix for duplicate library issues

import streamlit as st   # Web app framework
from PyPDF2 import PdfReader   # For extracting text from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter   # To split text into chunks
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings  # Convert text → vectors
from langchain_community.vectorstores import FAISS   # Vector database for similarity search
from dotenv import load_dotenv   # Load API keys securely
import google.generativeai as genai   # Google Gemini API
from datetime import datetime
import json
from functools import lru_cache

# ✅ Load API key for Gemini once at startup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------------------
# 🔹 Initialize session states
# ------------------------------
def init_session_state():
    """Keeps track of chat, memory, embeddings, and summaries"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {}
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# ------------------------------
# 🔹 Display previous chat history
# ------------------------------
def display_chat_history():
    """Show user and assistant messages in chat format"""
    for message in reversed(st.session_state.chat_history):
        with st.container():
            if message["role"] == "user":
                st.write(f"🧑 **You:** {message['content']}")
            else:
                st.write(f"🤖 **Assistant:** {message['content']}")

# ------------------------------
# 🔹 Manage conversation memory
# ------------------------------
class ConversationManager:
    def __init__(self, max_memory=5):
        self.max_memory = max_memory
    
    def add_to_memory(self, question, answer):
        """Save latest QnA in memory (keep only last 5)"""
        memory_item = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer
        }
        st.session_state.conversation_memory.append(memory_item)
        if len(st.session_state.conversation_memory) > self.max_memory:
            st.session_state.conversation_memory.pop(0)
    
    @staticmethod
    def get_relevant_memory(current_question, threshold=0.2):
        """Find relevant past questions using word overlap"""
        current_words = set(current_question.lower().split())
        current_word_count = len(current_words)
        
        relevant_memories = []
        for memory in st.session_state.conversation_memory:
            memory_words = set(memory['question'].lower().split())
            overlap = len(current_words.intersection(memory_words))
            if overlap / current_word_count >= threshold:
                relevant_memories.append(memory)
        
        return relevant_memories

# ------------------------------
# 🔹 PDF Processing
# ------------------------------
class PDFProcessor:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        # Split long PDF text into smaller chunks for embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        """Extract raw text from PDF file"""
        pdf_reader = PdfReader(pdf_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)

    def read_pdf(self, pdf_docs):
        """Read multiple PDFs and create summaries"""
        all_text = []
        for pdf in pdf_docs:
            doc_text = self.extract_text_from_pdf(pdf)
            all_text.append(doc_text)
            
            # Summarize each document
            summary = self.generate_document_summary(doc_text[:2000], pdf.name)
            st.session_state.document_summaries[pdf.name] = summary
        
        return ' '.join(all_text)

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_document_summary(text_preview, doc_name):
        """Generate summary using Gemini AI with caching"""
        try:
            prompt = f"""
            Please provide a concise summary of the following document:
            
            {text_preview}... [truncated]
            """
            
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def process_text(self, text):
        """Split text into manageable chunks"""
        return self.text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        """Convert text chunks → embeddings and store in FAISS"""
        try:
            vector_store = FAISS.from_texts(
                text_chunks, 
                embedding=st.session_state.embeddings
            )
            vector_store.save_local("faiss_db")
            return True
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    @staticmethod
    def get_gemini_response(context, question, relevant_memories=None):
        """Generate answer using PDF context + past memory"""
        try:
            memory_context = ""
            if relevant_memories:
                memory_context = "\nRelevant past conversations:\n" + "\n".join(
                    [f"Q: {m['question']}\nA: {m['answer']}" for m in relevant_memories[:2]]
                )

            context_text = "\n".join([doc.page_content for doc in context])

            prompt = f"""
            Context: {context_text}
            {memory_context}
            Question: {question}
            
            Answer using context + memory. If not found, reply "ANSWER_NOT_FOUND".
            """

            model = genai.GenerativeModel("models/gemini-1.5-flash")
            initial_response = model.generate_content(prompt)
            
            if initial_response.text.strip() == "ANSWER_NOT_FOUND":
                return "ANSWER_NOT_FOUND"
            
            return initial_response.text
        
        except Exception as e:
            raise Exception(f"Error getting Gemini response: {str(e)}")

# ------------------------------
# 🔹 Fallback: General Knowledge
# ------------------------------
def get_general_knowledge_response(question):
    """If answer not in PDFs, ask Gemini general knowledge"""
    try:
        prompt = f"Question: {question}\nGive a beginner-friendly answer."
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error getting general knowledge base response: {str(e)}")

# ------------------------------
# 🔹 Chat UI
# ------------------------------
def display_chat_interface():
    """Chat interface for user QnA"""
    st.write("💬 Chat with your PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        try:
            processor = PDFProcessor()
            relevant_memories = ConversationManager.get_relevant_memory(user_question)
            
            # Load FAISS vector store
            vector_store = FAISS.load_local(
                "faiss_db", 
                st.session_state.embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            context = retriever.invoke(user_question)
            
            with st.spinner("Thinking..."):
                response = processor.get_gemini_response(context, user_question, relevant_memories)
                
                # If not found in PDFs → fallback
                if response == "ANSWER_NOT_FOUND":
                    st.warning("Not found in PDFs. Do you want an AI answer?")
                    if st.button("Yes, please answer"):
                        general_response = get_general_knowledge_response(user_question)
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": general_response}
                        ])
                        processor.conversation_manager.add_to_memory(user_question, general_response)
                else:
                    # Store QnA in history + memory
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": user_question},
                        {"role": "assistant", "content": response}
                    ])
                    processor.conversation_manager.add_to_memory(user_question, response)
            
            display_chat_history()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ------------------------------
# 🔹 Main App
# ------------------------------
def main():
    st.set_page_config(page_title="Enhanced PDF Chat", layout="wide")
    st.header("📚 Advanced Exam Preparation with AI")
    
    init_session_state()
    processor = PDFProcessor()
    
    # Sidebar for file uploads + summaries
    with st.sidebar:
        tab1, tab2 = st.tabs(["📁 Upload", "📑 Summaries"])
        
        # File upload tab
        with tab1:
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
            
            if st.button("🔄 Process Documents"):
                if pdf_docs:
                    with st.spinner("Processing documents..."):
                        try:
                            raw_text = processor.read_pdf(pdf_docs)
                            text_chunks = processor.process_text(raw_text)
                            processor.create_vector_store(text_chunks)
                            st.success("✅ Documents processed successfully!")
                            st.session_state.processed_docs = [doc.name for doc in pdf_docs]
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Please upload at least one PDF file")
        
        # Document summaries tab
        with tab2:
            if st.session_state.document_summaries:
                for doc_name, summary in st.session_state.document_summaries.items():
                    with st.expander(f"Summary: {doc_name}"):
                        st.write(summary)
            else:
                st.info("No document summaries available yet.")

    # Display chat section
    display_chat_interface()
    
    # Export conversation option
    if st.session_state.chat_history:
        if st.button("📥 Export Conversation"):
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "chat_history": st.session_state.chat_history,
                "document_summaries": st.session_state.document_summaries
            }
            st.download_button(
                label="💾 Save Conversation",
                data=json.dumps(conversation_data, indent=2),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
