# 📚 Exam Preparation context QnA Bot with Gemini & FAISS :

Chat with your PDFs like they’re alive — upload lecture notes, textbooks, resumes, or any PDF, and ask questions in natural language. The app retrieves context from your documents and combines it with Gemini’s reasoning to answer clearly.

---

## 🚀 Features

* 📄 Upload and process **multiple PDFs**.
* ✂️ **Smart text chunking** for long documents.
* 🧠 **Vector embeddings with FAISS** for semantic search.
* 🤖 **Google Gemini integration** for natural QnA and summarization.
* 📝 **Document summaries** generated automatically.
* 💬 **Conversation memory** to keep context from past questions.
* 📥 Export your entire chat history + document summaries as JSON.

---

## 🛠️ Tech Stack

* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **Document parsing:** [PyPDF2](https://pypi.org/project/PyPDF2/)
* **Chunking:** `RecursiveCharacterTextSplitter` (LangChain)
* **Embeddings:** SpaCy (`en_core_web_sm`)
* **Vector DB:** [FAISS](https://faiss.ai/)
* **LLM:** Google Gemini (`gemini-1.5-flash`)
* **Config & Env:** python-dotenv
* **Persistence:** JSON export

---

## 🏗️ Architecture

```
        ┌───────────────┐
        │     PDF(s)    │
        └───────┬───────┘
                │ Extract (PyPDF2)
                ▼
        ┌────────────────────┐
        │ Text Splitter      │
        │ (chunk_size=1000)  │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Embeddings (SpaCy) │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Vector DB (FAISS)  │
        └─────────┬──────────┘
                  │ Retrieval (top-k)
                  ▼
        ┌─────────────────────────┐
        │ Gemini LLM              │
        │  - Uses context chunks  │
        │  - Falls back to GK if  │
        │    no match found       │
        └─────────┬──────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Chat UI (Streamlit)│
        │ + conversation mem │
        └────────────────────┘
```

---

## ⚙️ Installation & Setup

1. **Clone repo**

   ```bash
   git clone https://github.com/your-repo/pdf-qna-bot.git
   cd pdf-qna-bot
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set environment variable**
   Create a `.env` file in project root:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

6. **Run app**

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Usage

1. Upload one or more PDFs from the sidebar.
2. Click **"Process Documents"** — text is split, embedded, and stored in FAISS.
3. Ask any question in the chat box.

   * If answer is in PDF → retrieved and answered using Gemini.
   * If not found → Gemini provides a fallback general knowledge answer.
4. See **summaries per document** in the sidebar.
5. Export chat + summaries as JSON.

---

Made with ❤️‍🔥 in AI domain.
