import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os

from htmlTemplates import css, bot_template, user_template  # Ensure this file exists

# Function to initialize the Groq chat model
def init_groq_model():
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)  # Use st.secrets instead of os.getenv
    if not groq_api_key:
        st.warning("GROQ API Key is missing! Please set it in Streamlit secrets.")
        return None  # Return None if the key is not available

    return ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.2
    )

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Function to create conversation chain
def get_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

# Function to handle user input and respond
def handle_userinput(user_question):
    if "conversation" in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process your resume first.")

# Main function to run the Streamlit app
def main():
    load_dotenv()  # Load environment variables (for local use)

    st.set_page_config(page_title="Chat with Your Job Assistant", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Initialize LLM only once
    if "llm_groq" not in st.session_state:
        st.session_state.llm_groq = init_groq_model()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with Your Job Assistant 📚")
    user_question = st.text_input("Ask a question about your resume or job:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your resume PDFs here and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if pdf_docs:
                if st.session_state.llm_groq is None:
                    st.error("GROQ API Key is missing! Please configure it before proceeding.")
                    return

                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text found in the uploaded PDF. Please upload a valid resume.")

                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)

                        st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.llm_groq)
                        st.success("Resume processing complete! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
            else:
                st.warning("Please upload at least one resume PDF before processing.")

if __name__ == '__main__':
    main()

