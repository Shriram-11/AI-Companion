from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document  # Import Document
from dotenv import load_dotenv
import chromadb
import time
import random
import hashlib
from fastapi import FastAPI, HTTPException, Request, UploadFile, File  
import logging  
import uvicorn  
import traceback
from typing import List
from starlette.requests import Request
from starlette.responses import JSONResponse
import json
from fastapi.middleware.cors import CORSMiddleware
# load emvironment variables
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
# one thing i noticed if you device varables have same names as the .env files it will get the device variables
genai.configure(api_key=api_key)
# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Configure ChromaDB with local storage path
# clinet will change for cloud storage
client = chromadb.PersistentClient(path="./chroma_db")  # Specify the directory for database files
# Create or access a collection
#client.delete_collection("docs_collection")
collection = client.get_or_create_collection("docs_collection")

def get_text_from_pdf(file_path):
    """
    Reads text from a PDF file.
    Args:
        file_path: str - The path to the PDF file.
    Returns:
        str - The text from the PDF file.
    """
    try:
        pdf = PdfReader(file_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def get_text_from_txt(file_path):
    """
    Reads text from a TXT file.
    Args:
        file_path: str - The path to the TXT file.
    Returns:
        str - The text from the TXT file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return None

def get_chunks(text):
    """
    Splits text into chunks.
    Args:
        text: str - The text to split.
    Returns:
        list[str] - A list of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return None

def generate_doc_id():
    """
    Generates a unique, compact ID  to store vectors in the database.
    Args:
        None
    Returns:
        str - A unique ID
    """
    try:
        timestamp = int(time.time()).to_bytes(4, 'big')  # 4 bytes of UNIX timestamp
        random_part = random.getrandbits(48).to_bytes(6, 'big')  # 6 bytes of randomness
        unique_bytes = timestamp + random_part  # 10 bytes total 
        
        return hashlib.md5(unique_bytes).hexdigest()  # Full-length MD5 hash (32 chars)
    except Exception as e:
        print(f"Error generating doc ID: {e}")
        return None

def store_vectors(chunks):
    """
    IMPORTANT:
    The ChromaDB collection's dimensionality is fixed at the time of creation, based on the first inserted embeddings.
    For example, if the collection was initially populated with 384-dimensional embeddings, it will always expect 384 dimensions.
    If you change the embedding model or modify its configuration (such as using a model that outputs 768-dimensional vectors),
    this will cause a dimensionality mismatch error (e.g., "Embedding dimension 768 does not match collection dimensionality 384").

    In future:
        - DO NOT change the embedding model or its output dimensionality once the collection has been created.
        - If a change is necessary, you must either delete the previous data (or create a new collection) to match the new dimensionality,
        which may not be feasible in a production environment.
        
    To retain full information and avoid having to delete existing data, ensure that the embedding model and its configuration remain consistent.
    """
    """
    Args:
        chunks: list[str] - A list of text chunks to store in the database.
    Returns:
        None
    """
    try:
        # Create an instance of the embeddings class using the correct model
        embeddings_obj = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Generate embeddings for the list of texts using embed_documents
        vectors = embeddings_obj.embed_documents(chunks)
        
        # Iterate through the texts and store each one with its corresponding vector
        for i, text in enumerate(chunks):
            doc_id = generate_doc_id()  # Generate a unique ID for this text
            collection.add(
                ids=[doc_id],
                embeddings=[vectors[i]],  # vector for this text
                documents=[text]            # original text (optional)
            )
    except Exception as e:
        print(f"Error storing vectors: {e}")
        return None    

def process_files(files):
    """
    Process files to store vectors in the database.
    Args:
        files: list[str] - A list of file paths to process.
    Returns:
        bool - True if successful, False otherwise.
    """
    # Yet to be tested
    try:
        for file in files:
            if file.endswith(".pdf"):
                text=get_text_from_pdf(file)
                if text is None:
                    print("Error reading PDF:",file)
                    continue
            elif file.endswith(".txt"):
                text=get_text_from_txt(file)
                if text is None:
                    print("Error reading TXT:",file)
                    continue
            else:
                print("File type not supported")
                continue
            chunks=get_chunks(text)# get chunks
            store_vectors(chunks)# store vectors
        return True
    except Exception as e:
        print(f"Error processing files: {e}")
        return False

def get_conversational_chain():
    """
    Loads the conversational chain with Gemini API
    Args:
        None
    Returns:
        chain: function - The conversational chain function.
    """
    prompt_template = """
    You are a knowledgeable assistant. Use the provided context to answer the question in detail. 
    If the answer is not in the context, say: 'The answer is not available in the provided context,' 
    then continue answering based on what you know.'

    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Updated chain_type

    return chain

def ask_llm(user_prompt):
    """
    Queries the LLM after retrieving relevant documents
    Args:
        user_prompt: str - The user's question prompt.
    Returns:
        response: str - The LLM's response.
    """
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Ensure this model exists

        # Initialize ChromaDB client and collection
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="docs_collection")

        # Perform similarity search
        results = collection.query(
            query_embeddings=[embeddings.embed_query(user_prompt)],
            n_results=1  # Adjusted to 1 for simplicity
        )

        # Check if documents are retrieved
        if not results["documents"]:
            return {"output_text": "No relevant documents found."}

        # Load conversational chain
        chain = get_conversational_chain()
        
        # Retrieve documents and convert them to Document objects
        docs = [
            Document(page_content=doc) for doc in results["documents"][0]
        ]

        # Generate response
        response = chain.invoke(
            {"input_documents": docs, "question": user_prompt},
            return_only_outputs=True
        )

        return response
    except Exception as e:
        print(f"Error asking LLM: {e}")
        return {"output_text": "An error occurred while processing your request."}

# Example GET route
@app.get("/status")
async def health_check():

    return {"message": "API is running successfully", "status_code": 200}

# Example POST route
@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        file_paths = []

        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            
            # Save the file asynchronously
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            file_paths.append(file_path)

        # Process files
        success = process_files(file_paths)

        # Remove files after processing
        for file_path in file_paths:
            os.remove(file_path)

        if success:
            return {"message": "Success", "data": file_paths, "status_code": 200}
        else:
            return {"message": "Failure", "data": None, "status_code": 500}

    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/ask")
async def get_user_prompt(request: Request):
    try:
        body = await request.body()
        if not body:
            return JSONResponse({"error": "Empty request body"}, status_code=400)
        data = await request.json()
        if not data:
            raise ValueError("Empty JSON payload.")
        prompt = data.get("user_prompt", "")
        logger.info(f"User prompt received: {prompt}")
        prompt = str(prompt)
        if not prompt:
            raise ValueError("No user prompt provided.")
        
        result = ask_llm(prompt)
        if result is None:
            return {"message": "Failure", "data": None, "status_code": 500}
        else:
            return {"message": "Success", "data": {"output_text": result}, "status_code": 200}

    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except ValueError as ve:
        error_traceback = traceback.format_exc()
        logger.error(f"User prompt failed: {str(ve)}\nStack trace: {error_traceback}")
        logger.error(f"User prompt failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Bad Request: {str(ve)}")
    except Exception as e:
        logger.error(f"User prompt failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def check_gemini_llm():
    """
    Checks if the Gemini LLM is responding.
    Args:
        None
    Returns:
        bool - True if the LLM is responding, False otherwise.
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key, temperature=0.3)
        response = model.invoke("Hello, are you there?")
        return response is not None
    except Exception as e:
        print(f"Error checking Gemini LLM: {e}")
        return False

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Args:
        None
    Returns:
        dict - The health status.
    """
    return {"status": "UP"}

@app.get("/health_status")
def health_status():
    """
    Health status endpoint.
    Args:
        None
    Returns:
        dict - The health status.
    """
    return {"status": "UP"}

if __name__ == "__main__":
    if check_gemini_llm():
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)  # Ensure host and port are correct
    else:
        print("Gemini LLM is not responding. Please check the configuration and try again.")