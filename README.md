# PDF Question Answering System

## Features
- Upload any PDF file
- Ask questions in natural language
- AI gives answers based on PDF content

## How to Run
1. Install Python
2. Run: pip install -r requirements.txt
3. Run: streamlit run app.py
4. Upload PDF and ask questions

## Technologies Used
- Streamlit (UI)
- LangChain (PDF processing)
- HuggingFace Transformers (AI model)
- FAISS (Vector search)

## Sample Output
Upload: contract.pdf
Question: "What is the payment amount?"
Answer: "$5000 USD"