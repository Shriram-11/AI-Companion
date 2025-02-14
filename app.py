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
import json

# load emvironment variables
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
# one thing i noticed if you device varables have same names as the .env files it will get the device variables
genai.configure(api_key=api_key)

# Configure ChromaDB with local storage path
# clinet will change for cloud storage
client = chromadb.PersistentClient(path="./chroma_db")  # Specify the directory for database files
# Create or access a collection
#client.delete_collection("docs_collection")
collection = client.get_or_create_collection("docs_collection")

def get_text_from_pdf(file_path):
    """
    Reads text from a PDF file.
    :param file_path: The path to the PDF file.
    :return: The text from the PDF file
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
    :param file_path: The path to the TXT file.
    :return: The text from the TXT file
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
    :param text (str): The text to split.
    :rtype-list[str]: A list of text chunks.
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
    rtype: str - A unique ID
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
        None
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
    except Exception as e:
        print(f"Error processing files: {e}")
        return None

def get_conversational_chain():
    """Loads the conversational chain with Gemini API"""
    prompt_template = """
   You are a knowledgeable assistant. Use the provided context to answer the question in detail. If the answer is not in the context, say: 'The answer is not available in the provided context,' then continue answering based on what you know.'
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",api_key="KEY",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def ask_llm(user_prompt):
    """Queries the LLM after retrieving relevant documents"""
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Ensure this model exists

    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="docs_collection")

    # Perform similarity search
    results = collection.query(
        query_embeddings=[embeddings.embed_query(user_prompt)],
        n_results=1  
    )

    print(results)

    # Load conversational chain
    chain = get_conversational_chain()
    
    # Retrieve documents and convert them to Document objects
    docs = [
        Document(page_content=doc) for doc in results["documents"][0]
    ] if results["documents"] else []

    # Generate response
    response = chain(
        {"input_documents": docs, "question": user_prompt},
        return_only_outputs=True
    )

    return response

