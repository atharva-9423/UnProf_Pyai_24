import os
import sys
import tempfile
import warnings

# Suppress HuggingFace and warning messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 🔐 Basic API Key Authentication (Hardcoded for demonstration)
API_KEY = "my-secret-key"

app = FastAPI(
    title="Complete RAG API",
    description="A complete FastAPI REST endpoint that supports dynamic document uploads (PDF/TXT) and Conversational RAG.",
    version="2.0.0"
)

# Global variables
vectorstore = None
rag_chain = None
text_splitter = None
embeddings = None

# Pydantic models for structured JSON request/response
class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[List[str]]] = []

class ChatResponse(BaseModel):
    answer: str
    chat_history: List[List[str]]

class UploadResponse(BaseModel):
    message: str
    chunks_added: int

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.on_event("startup")
def startup_event():
    global vectorstore, rag_chain, text_splitter, embeddings
    print("Initializing RAG Pipeline on Server Startup...")

    if "GEMINI_API_KEY" not in os.environ:
        print("\nWARNING: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    # 1. Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Initialize an empty FAISS vectorstore with a tiny welcome document
    # We need at least one document to initialize the FAISS schema properly.
    from langchain_core.documents import Document
    initial_doc = Document(page_content="Welcome to the RAG API. This is the initial system document. I can process any documents you upload.")
    vectorstore = FAISS.from_documents([initial_doc], embeddings)

    # 3. Initialize the LCEL Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    def get_contextualized_question(inputs):
        history = inputs.get("chat_history", [])
        if not history:
            return inputs["input"]
        
        last_human_query = ""
        for role, text in reversed(history):
            if role == "human":
                last_human_query = text
                break
        return f"{last_human_query} {inputs['input']}"

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Context: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=get_contextualized_question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG Pipeline Ready (Waiting for uploads)!")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    Upload a .txt or .pdf document to dynamically index it into the vector database.
    """
    global vectorstore, text_splitter
    
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")
    
    # Save the uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        
        # Load the document
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file.name)
        else:
            loader = TextLoader(temp_file.name)
            
        docs = loader.load()
        
        # Chunk the document
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            raise HTTPException(status_code=400, detail="Could not extract text from the file.")
            
        # Add chunks to the existing in-memory vectorstore
        vectorstore.add_documents(splits)
        
        return UploadResponse(
            message=f"Successfully indexed '{file.filename}'",
            chunks_added=len(splits)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """
    Query the RAG chatbot. The chatbot will search through any previously uploaded documents.
    """
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

    try:
        history_tuples = [(item[0], item[1]) for item in request.chat_history]
        
        answer = rag_chain.invoke({
            "input": request.query,
            "chat_history": history_tuples
        })

        updated_history = request.chat_history.copy()
        updated_history.append(["human", request.query])
        updated_history.append(["ai", answer])

        return ChatResponse(answer=answer, chat_history=updated_history)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
