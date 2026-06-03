# app.py - ULTIMATE PDF QA SYSTEM
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import faiss
import torch
from datetime import datetime
import re

st.set_page_config(
    page_title="Smart PDF QA System", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 18px !important;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title with animation effect
st.title("📚 Smart PDF Question Answering System")
st.markdown("*Powered by Google Flan-T5 & Sentence Transformers*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_size = st.selectbox(
        "Model Size (Quality vs Speed)",
        ["small (fast)", "base (balanced)", "large (best quality)"]
    )
    
    model_map = {
        "small (fast)": "google/flan-t5-small",
        "base (balanced)": "google/flan-t5-base",
        "large (best quality)": "google/flan-t5-large"
    }
    selected_model = model_map[model_size]
    
    # Chunk size
    chunk_size = st.slider("Context Chunk Size", 200, 1000, 500)
    
    # Number of chunks to retrieve
    top_k = st.slider("Number of relevant chunks", 1, 5, 3)
    
    # Temperature for creativity
    temperature = st.slider("Answer Creativity", 0.0, 1.0, 0.3)
    
    st.divider()
    st.markdown("### 📊 Features")
    st.markdown("✅ Semantic Search")
    st.markdown("✅ Smart Summarization")
    st.markdown("✅ Conversation History")
    st.markdown("✅ Export Results")

# Cache for conversation history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'summary' not in st.session_state:
    st.session_state.summary = None

@st.cache_resource
def load_models(model_name):
    with st.spinner("🚀 Loading AI models (first time may take 1-2 minutes)..."):
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return embedder, tokenizer, model

def clean_text(text):
    """Clean extracted text"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def smart_chunking(text, chunk_size):
    """Smart chunking by sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) < chunk_size:
            current_chunk += sent + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_summary(text, tokenizer, model, style="concise"):
    """Generate summary with different styles"""
    if style == "concise":
        prompt = f"Summarize this text in 2-3 sentences: {text[:1500]}"
        max_tokens = 150
    elif style == "detailed":
        prompt = f"Provide a detailed summary of this text in 5-7 sentences: {text[:1500]}"
        max_tokens = 300
    else:  # bullet points
        prompt = f"Summarize this text as bullet points: {text[:1500]}"
        max_tokens = 200
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_tokens, 
        temperature=0.5,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main area - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📄 Upload your PDF document", 
        type="pdf",
        help="Supports any PDF with selectable text"
    )
    
    # Mode selection with icons
    mode = st.radio(
        "🎯 Select Mode",
        ["❓ Ask Questions", "📝 Summarize Document", "💬 Chat with Document"],
        horizontal=True
    )

with col2:
    if uploaded_file:
        st.info(f"✅ **File loaded:** {uploaded_file.name}")
        st.caption(f"Size: {uploaded_file.size / 1024 / 1024:.2f} MB")

# Process PDF
if uploaded_file:
    with st.spinner("📖 Processing your PDF..."):
        # Extract text
        reader = PdfReader(uploaded_file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n" + page_text
        
        text = clean_text(text)
        
        if not text.strip():
            st.error("❌ No text could be extracted. Make sure the PDF is not scanned.")
            st.stop()
        
        # Smart chunking
        chunks = smart_chunking(text, chunk_size)
        
        # Load models
        embedder, tokenizer, model = load_models(selected_model)
        
        # Create embeddings
        embeddings = embedder.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        st.success(f"✅ Processed {len(chunks)} text chunks from {len(reader.pages)} pages")
        
        # ========== SUMMARIZE MODE ==========
        if mode == "📝 Summarize Document":
            st.subheader("📄 Document Summary")
            
            summary_style = st.selectbox(
                "Summary Style",
                ["concise", "detailed", "bullet points"]
            )
            
            if st.button("🔍 Generate Summary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(text, tokenizer, model, summary_style)
                    st.session_state.summary = summary
                    
                    st.markdown("---")
                    if summary_style == "bullet points":
                        st.markdown("### 📌 Key Points")
                        for point in summary.split('\n'):
                            if point.strip():
                                st.markdown(f"• {point}")
                    else:
                        st.markdown("### 📝 Summary")
                        st.write(summary)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        # ========== ASK QUESTIONS MODE ==========
        elif mode == "❓ Ask Questions":
            st.subheader("❓ Ask Anything")
            
            # Suggested questions
            with st.expander("💡 Suggested Questions"):
                st.markdown("- What is the main topic of this document?")
                st.markdown("- What are the key findings?")
                st.markdown("- Summarize the conclusion")
                st.markdown("- What problem does this solve?")
            
            question = st.text_input("Type your question here:", placeholder="e.g., What is this document about?")
            
            if question and st.button("🔍 Get Answer", type="primary"):
                with st.spinner("🧠 Finding answer in document..."):
                    # Semantic search
                    q_embed = embedder.encode([question])
                    distances, indices = index.search(
                        np.array(q_embed).astype('float32'), 
                        min(top_k, len(chunks))
                    )
                    
                    # Get context with page numbers
                    contexts = []
                    for i in indices[0]:
                        chunk_text = chunks[i]
                        # Extract page number if available
                        page_match = re.search(r'--- Page (\d+) ---', chunk_text)
                        if page_match:
                            page_num = page_match.group(1)
                            chunk_text = re.sub(r'--- Page \d+ ---', '', chunk_text)
                            contexts.append(f"[Page {page_num}] {chunk_text.strip()}")
                        else:
                            contexts.append(chunk_text)
                    
                    context = " ".join(contexts)
                    
                    # Generate answer
                    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context: {context}

Question: {question}

Provide a clear, accurate answer:"""
                    
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True)
                    outputs = model.generate(
                        inputs.input_ids, 
                        max_new_tokens=200, 
                        temperature=temperature,
                        do_sample=temperature > 0,
                        top_p=0.9
                    )
                    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean answer
                    answer = answer.replace(prompt, "").strip()
                    
                    # Store in history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Display answer
                    st.markdown("---")
                    st.markdown("### ✅ Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Show confidence and sources
                    with st.expander("📖 Source Text & Confidence"):
                        st.caption(f"Retrieved {len(indices[0])} relevant sections")
                        for i, ctx in enumerate(contexts):
                            st.text_area(f"Source {i+1}", ctx[:300], height=100)
        
        # ========== CHAT MODE ==========
        else:  # Chat mode
            st.subheader("💬 Chat with Your Document")
            
            # Display chat history
            for chat in st.session_state.chat_history[-10:]:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    st.caption(chat["timestamp"])
            
            # Chat input
            question = st.chat_input("Ask a question about your document...")
            
            if question:
                # Add user message
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Semantic search
                        q_embed = embedder.encode([question])
                        distances, indices = index.search(
                            np.array(q_embed).astype('float32'), 
                            min(top_k, len(chunks))
                        )
                        
                        context = " ".join([chunks[i] for i in indices[0]])
                        
                        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                        outputs = model.generate(inputs.input_ids, max_new_tokens=200, temperature=0.3)
                        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        answer = answer.replace(prompt, "").strip()
                        
                        st.write(answer)
                        
                        # Store in history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
        
        # ========== EXPORT SECTION ==========
        if st.session_state.chat_history and mode != "📝 Summarize Document":
            st.divider()
            if st.button("📥 Export Chat History", use_container_width=True):
                export_text = "\n\n".join([
                    f"Q: {chat['question']}\nA: {chat['answer']}" 
                    for chat in st.session_state.chat_history
                ])
                st.download_button(
                    label="Download Chat",
                    data=export_text,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )

else:
    # Welcome screen
    st.info("👈 Please upload a PDF document to get started")
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload PDF** - Click the upload button and select your PDF
        2. **Choose Mode** - Ask questions, get summary, or chat with document
        3. **Adjust Settings** - Use sidebar to control quality vs speed
        4. **Get Answers** - Ask anything about your document
        
        **Tips:**
        - For best results, use PDFs with selectable text (not scanned)
        - Larger models give better answers but take longer
        - Adjust chunk size for better context
        """)