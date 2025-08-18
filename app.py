import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import google.generativeai as genai
from datetime import datetime
import json
from functools import lru_cache

# Load environment variables once at startup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize session states
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {}
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def display_chat_history():
    """Display the chat history with user and assistant messages"""
    for message in reversed(st.session_state.chat_history):
        with st.container():
            if message["role"] == "user":
                st.write(f"🧑 **You:** {message['content']}")
            else:
                st.write(f"🤖 **Assistant:** {message['content']}")

class ConversationManager:
    def __init__(self, max_memory=5):
        self.max_memory = max_memory
    
    def add_to_memory(self, question, answer):
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
        """Optimized relevance check using word set operations"""
        current_words = set(current_question.lower().split())
        current_word_count = len(current_words)
        
        relevant_memories = []
        for memory in st.session_state.conversation_memory:
            memory_words = set(memory['question'].lower().split())
            overlap = len(current_words.intersection(memory_words))
            if overlap / current_word_count >= threshold:
                relevant_memories.append(memory)
        
        return relevant_memories

class PDFProcessor:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        """Extract text from a single PDF file"""
        pdf_reader = PdfReader(pdf_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)

    def read_pdf(self, pdf_docs):
        """Process multiple PDF documents in parallel"""
        all_text = []
        for pdf in pdf_docs:
            doc_text = self.extract_text_from_pdf(pdf)
            all_text.append(doc_text)
            
            # Generate and store document summary
            summary = self.generate_document_summary(doc_text[:2000], pdf.name)
            st.session_state.document_summaries[pdf.name] = summary
        
        return ' '.join(all_text)

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_document_summary(text_preview, doc_name):
        """Generate document summary with caching"""
        try:
            prompt = f"""
            Please provide a concise summary of the following document:
            
            {text_preview}... [truncated]
            
            Include:
            1. Main topics covered
            2. Key points
            3. Document type and structure
            
            Limit to 200 words.
            """
            
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
        """Generate document summary with error handling and text truncation"""
        try:
            # Clean and truncate text to avoid token limits
            cleaned_text = ' '.join(text_preview.split())  # Remove extra whitespace
            truncated_text = cleaned_text[:1000] if len(cleaned_text) > 1000 else cleaned_text
            
            prompt = f"""
            Summarize the following document excerpt in 2-3 sentences:
            {truncated_text}
            """
            
            model = genai.GenerativeModel("models/gemini-1.5-flash")

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
            
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 150,
                }
            )
            
            if response.text:
                return response.text.strip()
            else:
                return f"Summary generation failed for {doc_name}"
                
        except Exception as e:
            return f"Document loaded successfully. Summary unavailable: {str(e)}"
    def process_text(self, text):
        """Process text with optimized chunking"""
        return self.text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        """Create and save vector store"""
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
   # """Get AI response with fallback to general knowledge base"""
     try:
        memory_context = ""
        if relevant_memories:
            memory_context = "\nRelevant past conversations:\n" + "\n".join(
                [f"Q: {m['question']}\nA: {m['answer']}" for m in relevant_memories[:2]]
            )

        # Extract text from context documents
        context_text = "\n".join([doc.page_content for doc in context])

        # First attempt to answer from context
        prompt = f"""
        Context: {context_text}
        {memory_context}
        Question: {question}
        
        Provide a detailed answer using the context and past conversations, Include all references from context when applicable. with best readability presentation
        If you cannot find the answer from the context or past conversations, respond with exactly "ANSWER_NOT_FOUND".
        """

        model = genai.GenerativeModel("models/gemini-1.5-flash")
        initial_response = model.generate_content(prompt)
        
        if initial_response.text.strip() == "ANSWER_NOT_FOUND":
            return "ANSWER_NOT_FOUND"
        
        return initial_response.text
        
     except Exception as e:
        raise Exception(f"Error getting Gemini response: {str(e)}")
    
def get_general_knowledge_response(question):
    """Get response from Gemini's  knowledge"""
    try:
        prompt = f"""
        Question: {question}
        
        Please provide a precised answer based on your knowledge and web search with best readability presentation such that a beginner can understand easily.
        """
        
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error getting general knowledge base response: {str(e)}")

def display_chat_interface():
    """Handle chat interface and user interaction"""
    st.write("💬 Chat with your PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        try:
            processor = PDFProcessor()
            relevant_memories = ConversationManager.get_relevant_memory(user_question)
            
            vector_store = FAISS.load_local(
                "faiss_db", 
                st.session_state.embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            context = retriever.invoke(user_question)
            
            with st.spinner("Thinking..."):
                response = processor.get_gemini_response(context, user_question, relevant_memories)
                
                if response == "ANSWER_NOT_FOUND":
                    st.warning("I couldn't find the answer in the provided documents. Would you like me to answer based on my knowledge base?")
                    if st.button("Yes, please answer"):
                        with st.spinner("Generating response..."):
                            general_response = get_general_knowledge_response(user_question)
                            st.session_state.chat_history.extend([
                                {"role": "user", "content": user_question},
                                {"role": "assistant", "content": "I couldn't find this information in the documents, but based on my knowledge base:\n\n" + general_response}
                            ])
                            processor.conversation_manager.add_to_memory(user_question, general_response)
                else:
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": user_question},
                        {"role": "assistant", "content": response}
                    ])
                    processor.conversation_manager.add_to_memory(user_question, response)
            
            display_chat_history()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(page_title="Enhanced PDF Chat", layout="wide")
    st.header("📚 Advanced Exam Preparation with AI")
    
    init_session_state()
    processor = PDFProcessor()
    
    with st.sidebar:
        tab1, tab2 = st.tabs(["📁 Upload", "📑 Summaries"])
        
        with tab1:
            pdf_docs = st.file_uploader(
                "Upload your PDF Files",
                accept_multiple_files=True,
                type=['pdf']
            )
            
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
        
        with tab2:
            if st.session_state.document_summaries:
                for doc_name, summary in st.session_state.document_summaries.items():
                    with st.expander(f"Summary: {doc_name}"):
                        st.write(summary)
            else:
                st.info("No document summaries available yet.")

    display_chat_interface()
    
    # Export functionality
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