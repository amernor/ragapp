import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="Moondream2 Vision RAG", page_icon="🌙", layout="wide")

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from manage_vectordb import VectorDB
import tempfile
import os
import time
import torch
from PIL import Image

# Configuration from environment variables
model_name = os.getenv("MODEL_NAME", "vikhyatk/moondream2")
chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# PostgreSQL/pgvector configuration
db_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
db_port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "ragdb")
db_user = os.getenv("POSTGRES_USER", "postgres")
db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
collection_name = os.getenv("COLLECTION_NAME", "documents")

# Initialize VectorDB
vdb = VectorDB(
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_user,
    password=db_password,
    collection_name=collection_name,
    embedding_model=embedding_model
)

# Load Moondream2 model
@st.cache_resource
def load_model():
    """Load the Moondream2 model and tokenizer with aggressive memory optimization"""
    try:
        # Load with aggressive memory optimization for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision="2024-08-26",
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        # Set model to eval mode to save memory
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="2024-08-26")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Load model lazily - only when needed
if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['tokenizer'] = None
    st.session_state['model_loaded'] = False

def split_docs(raw_documents):
    """Split documents into chunks for embedding"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(raw_documents)
    return docs


def read_file(file):
    """Read uploaded PDF, text, or markdown file"""
    file_type = file.type
    file_name = file.name.lower()

    if file_type == "application/pdf":
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp.name, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp.name)

    elif file_type == "text/plain" or file_name.endswith('.txt'):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with open(temp.name, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp.name)

    elif file_type == "text/markdown" or file_name.endswith('.md'):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
        with open(temp.name, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp.name)

    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    raw_documents = loader.load()
    return raw_documents

def read_image(file):
    """Read uploaded image file"""
    return Image.open(file)


# Streamlit UI
st.title("🌙 Moondream2 Vision RAG Chatbot")
st.markdown("Upload documents and images, then ask questions about them!")

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Document Upload")

    files = st.file_uploader(
        label="Upload PDF, text, or markdown files",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
        help="Upload one or more documents to populate the knowledge base"
    )
    
    st.header("🖼️ Image Upload")
    
    image_file = st.file_uploader(
        label="Upload an image (optional)",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to analyze with vision capabilities"
    )

    if files:
        st.success(f"✅ Loaded {len(files)} file(s):")
        for f in files:
            st.write(f"  • {f.name}")

        # Add a button to process the files
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    all_documents = []

                    # Process each file
                    for file in files:
                        st.info(f"Processing {file.name}...")
                        raw_docs = read_file(file)
                        docs = split_docs(raw_docs)
                        all_documents.extend(docs)

                    st.info(f"Total: {len(all_documents)} chunks from {len(files)} file(s)")

                    # Populate the vector database with all documents
                    vdb.populate_db(all_documents)
                    st.session_state['db_populated'] = True
                    st.session_state['file_count'] = len(files)
                    st.session_state['file_names'] = [f.name for f in files]
                    st.success("Documents processed and ready for questions!")

                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    st.session_state['db_populated'] = False

    # Display uploaded image
    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        st.session_state['uploaded_image'] = read_image(image_file)
        st.success("✅ Image loaded!")
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This Vision RAG chatbot powered by **Moondream2**:
    - Supports **PDF, TXT, and Markdown** files
    - Analyzes **images** with vision-language model
    - Accepts **multiple file uploads**
    - Uses **pgvector** for document storage
    - Retrieves relevant context before answering
    """)

# Main chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Upload documents or images, then ask me questions. I can understand both text and images!"}
    ]

if 'db_populated' not in st.session_state:
    st.session_state['db_populated'] = False

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents or image..."):
    # Check if we have either documents or an image
    has_content = st.session_state.get('db_populated', False) or st.session_state.get('uploaded_image') is not None
    
    if not has_content:
        st.warning("⚠️ Please upload and process a document or upload an image first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    context_parts = []
                    
                    # Get document context if available
                    if st.session_state.get('db_populated', False):
                        retriever = vdb.get_retriever()
                        retrieved_docs = retriever.get_relevant_documents(prompt)
                        print(f"\n📚 Retrieved {len(retrieved_docs)} relevant chunks:")
                        for i, doc in enumerate(retrieved_docs, 1):
                            preview = doc.page_content[:100].replace('\n', ' ')
                            print(f"  {i}. {preview}...")
                            context_parts.append(doc.page_content)
                    
                    # Load model on first use
                    if st.session_state['model'] is None:
                        with st.spinner("Loading Moondream2 model (this may take a few minutes)..."):
                            st.session_state['model'], st.session_state['tokenizer'] = load_model()
                            if st.session_state['model'] is not None:
                                st.session_state['model_loaded'] = True
                    
                    model = st.session_state['model']
                    tokenizer = st.session_state['tokenizer']
                    
                    if model is None:
                        st.error("Model failed to load. Please check the logs.")
                        response = "Sorry, the model could not be loaded."
                    # Build the prompt for Moondream2
                    elif st.session_state.get('uploaded_image') is not None:
                        # If we have an image, use vision capabilities
                        image = st.session_state['uploaded_image']
                        
                        # Encode the image
                        enc_image = model.encode_image(image)
                        
                        # Add document context if available
                        full_prompt = prompt
                        if context_parts:
                            context_text = "\n\n".join(context_parts)
                            full_prompt = f"Context from documents:\n{context_text}\n\nQuestion about the image: {prompt}"
                        
                        # Generate response using Moondream2
                        response = model.answer_question(enc_image, full_prompt, tokenizer)
                    elif model is not None:
                        # Text-only mode
                        if context_parts:
                            context_text = "\n\n".join(context_parts)
                            full_prompt = f"Answer the question based on the following context:\n\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
                            
                            # Use the model as a text generator
                            inputs = tokenizer(full_prompt, return_tensors="pt")
                            with torch.no_grad():  # Save memory during inference
                                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            # Extract just the answer part
                            if "Answer:" in response:
                                response = response.split("Answer:")[-1].strip()
                        else:
                            response = "Please upload a document or image first."
                    else:
                        response = "Model not available."
                    
                    st.write(response)

                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    print(f"Full error: {e}")
                    import traceback
                    traceback.print_exc()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })