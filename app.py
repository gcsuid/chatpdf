import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""
    return text

def get_text_chunks(text, model_name):
    """Split text into chunks for processing"""
    try:
        if model_name == "Google AI":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, 
                chunk_overlap=1000
            )
        else:
            # Default text splitter for other models
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, 
                chunk_overlap=1000
            )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

def get_vector_store(text_chunks, model_name, api_key=None):
    """Create vector store from text chunks"""
    try:
        if model_name == "Google AI":
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )
        else:
            st.error("Unsupported model for vector store creation")
            return None
            
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    """Create conversational chain for Q&A"""
    try:
        if model_name == "Google AI":
            prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.3, 
                google_api_key=api_key
            )
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        else:
            st.error("Unsupported model for conversational chain")
            return None
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    """Process user input and generate response"""
    if not api_key:
        st.warning("Please provide API key before processing.")
        return
    
    if not pdf_docs:
        st.warning("Please upload PDF files before processing.")
        return
    
    try:
        # Extract text and create chunks
        pdf_text = get_pdf_text(pdf_docs)
        if not pdf_text:
            st.error("No text could be extracted from the PDFs.")
            return
            
        text_chunks = get_text_chunks(pdf_text, model_name)
        if not text_chunks:
            st.error("Could not split text into chunks.")
            return
        
        # Create vector store
        vector_store = get_vector_store(text_chunks, model_name, api_key)
        if not vector_store:
            return
        
        # Process question
        if model_name == "Google AI":
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )
            new_db = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain("Google AI", vectorstore=new_db, api_key=api_key)
            
            if not chain:
                return
                
            response = chain(
                {"input_documents": docs, "question": user_question}, 
                return_only_outputs=True
            )
            
            response_text = response['output_text']
            pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
            
            # Add to conversation history
            conversation_history.append((
                user_question, 
                response_text, 
                model_name, 
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                ", ".join(pdf_names)
            ))
            
            # Display current conversation
            display_conversation(user_question, response_text)
            
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def display_conversation(user_question, response_text):
    """Display the current question and answer"""
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
            </div>
            <div class="message">{response_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_conversation_history(conversation_history):
    """Display previous conversations"""
    # Skip the most recent conversation (already displayed)
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_download_link(conversation_history):
    """Create download link for conversation history"""
    if len(conversation_history) > 0:
        df = pd.DataFrame(
            conversation_history, 
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px; border: none;">Download Conversation History</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Chat with multiple PDFs", 
        page_icon=":books:",
        layout="wide"
    )
    st.header("Chat with multiple PDFs (v1) :books:")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = False

    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox("Select the Model:", ["Google AI"])
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Enter your Google API Key:", 
        type="password",
        help="Click [here](https://ai.google.dev/) to get an API key."
    )
    
    if not api_key:
        st.sidebar.warning("Please enter your Google API Key to proceed.")
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Clear Chat", help="Clear conversation history"):
        st.session_state.conversation_history = []
        st.rerun()
    
    if col2.button("Reset All", help="Reset everything"):
        st.session_state.conversation_history = []
        st.session_state.processed_pdfs = False
        # Clear FAISS index if it exists
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        st.rerun()

    # File upload
    st.sidebar.title("Upload Documents")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files", 
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if st.sidebar.button("Process Documents"):
        if pdf_docs and api_key:
            with st.spinner("Processing documents..."):
                try:
                    # Test the connection and processing
                    pdf_text = get_pdf_text(pdf_docs)
                    if pdf_text:
                        text_chunks = get_text_chunks(pdf_text, model_name)
                        if text_chunks:
                            vector_store = get_vector_store(text_chunks, model_name, api_key)
                            if vector_store:
                                st.session_state.processed_pdfs = True
                                st.sidebar.success("Documents processed successfully!")
                            else:
                                st.sidebar.error("Failed to create vector store.")
                        else:
                            st.sidebar.error("Failed to create text chunks.")
                    else:
                        st.sidebar.error("No text extracted from PDFs.")
                except Exception as e:
                    st.sidebar.error(f"Error processing documents: {str(e)}")
        else:
            if not pdf_docs:
                st.sidebar.warning("Please upload PDF files.")
            if not api_key:
                st.sidebar.warning("Please provide API key.")

    # Download conversation history
    create_download_link(st.session_state.conversation_history)

    # Main chat interface
    st.markdown("### Ask questions about your uploaded documents")
    
    # Display status
    if not api_key:
        st.info("👈 Please enter your Google API Key in the sidebar to get started.")
    elif not pdf_docs:
        st.info("👈 Please upload PDF files in the sidebar.")
    elif not st.session_state.processed_pdfs:
        st.info("👈 Please click 'Process Documents' in the sidebar after uploading PDFs.")
    else:
        st.success("✅ Ready to answer questions about your documents!")

    # Question input
    user_question = st.text_input(
        "Ask a question about your PDF documents:",
        placeholder="What is the main topic discussed in the document?",
        disabled=not (api_key and pdf_docs and st.session_state.processed_pdfs)
    )
    
    # Process question
    if user_question and api_key and pdf_docs and st.session_state.processed_pdfs:
        user_input(
            user_question, 
            model_name, 
            api_key, 
            pdf_docs, 
            st.session_state.conversation_history
        )
        
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        display_conversation_history(st.session_state.conversation_history)
        
        # Add some visual flair
        if len(st.session_state.conversation_history) > 0:
            st.balloons()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Upload PDF documents and ask questions to get detailed answers using AI.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()