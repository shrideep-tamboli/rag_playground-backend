from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the uploaded file content
uploaded_file_content = None

class FineTuning(BaseModel):
    llm: Optional[str] = None
    framework: Optional[str] = None
    textSplitter: Optional[str] = None
    embeddingModel: Optional[str] = None
    chunkSize: Optional[str] = None
    vectorStore: Optional[str] = None

class ProcessRequest(BaseModel):
    ragMethod1: Optional[str] = None
    ragMethod2: Optional[str] = None
    ragMethod3: Optional[str] = None
    fineTuning1: Optional[FineTuning] = None
    fineTuning2: Optional[FineTuning] = None
    fineTuning3: Optional[FineTuning] = None
    query: str

@app.get('/')
def get_data():
    return {"message": "Hello from FastAPI!"}

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    global uploaded_file_content, uploaded_file_name  # Use the global variable
    uploaded_file_content = await file.read()
    uploaded_file_name = file.filename  # Store the entire file content
    file_length = len(uploaded_file_content)
    print(f"Uploaded file size: {file_length} bytes")
    return {"filename": file.filename, "length": file_length}  # Return the uploaded file info

@app.post('/process')
async def process_query(request: ProcessRequest):
    received_data = []
    rag_methods = []
    rag_results = []
    results = []

    print("Received request:", request)
##
    # Access the global uploaded file content
    if uploaded_file_content is not None:
        received_data.append(f"Uploaded file size: {len(uploaded_file_content)} bytes")

    for i, method in enumerate([request.ragMethod1, request.ragMethod2, request.ragMethod3], start=1):
        if method:
            fine_tuning = getattr(request, f'fineTuning{i}')
            received_data.append(f"Received RAG method{i}: {method}")
            if fine_tuning:
                fine_tuning_details = fine_tuning.dict(exclude_none=True)
                fine_tuning_str = ", ".join(f"{k}={v}" for k, v in fine_tuning_details.items())
                received_data.append(f"Fine-tuning for method{i}: {fine_tuning_str}")
            rag_methods.append({
                "index": i, 
                "method": method, 
                "fine_tuning": fine_tuning.dict(exclude_none=True) if fine_tuning else None
            })
            
            # Call the appropriate RAG method function if file is uploaded
            if uploaded_file_content is None:
                result = "Upload a file"
            else:
                if method == "Traditional RAG":
                    result = vector_retrieval(method, request.query, uploaded_file_name, uploaded_file_content, fine_tuning)  # Pass file as an argument
                elif method == "Multi-modal RAG":
                    result = multi_modal_rag(method, request.query, fine_tuning)
                elif method == "Agentic RAG":
                    result = agentic_rag(method, request.query, fine_tuning)
                elif method == "Graph RAG":
                    result = graph_rag(method, request.query, fine_tuning)
                else:
                    result = None

            rag_results.append(result)  # Append the result for this method

    # Print the results after processing all methods
    for index, result in enumerate(rag_results):
        print(f"RAG_RESULTS[{index}]: {result}")
            #if results:
             #   rag_results.append({"method": method, "result": results})
              #  print("RAG RESULTS: ",rag_results) ####################### Print the RAG RESULTS HERE

    if request.query:
        received_data.append(f"Received query: {request.query}")
    
    for data in received_data:
        print(data)
    
    response = {
        "message": "Data received successfully",
        "query": request.query,
        "query_length": len(request.query),
        "rag_methods": rag_methods,
        "rag_results": rag_results,
        "uploaded_file_size": len(uploaded_file_content) if uploaded_file_content else None,  # Include uploaded file size
        #"file_content": file_content_str[:100]  # Include the first 100 characters of the file content
    }

    return response
#######################################################################################################

import os
from dotenv import load_dotenv
load_dotenv()
## Environment varibale initialization
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
groq_api_key = os.getenv("groq_api_key")
OPENAI_API_KEY2 = os.getenv('OPENAI_API_KEY2')
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="RAG_Playground"

#### INITIALIZE FT METHODS SENT BY USER



####  FRAMEWORK IS A FINETUNING VARIABLE LANGCHAIN HAS TO BE ENCAPSULATED IN IF FINE_TUNINING["FRAMEWORK":"LANGCHAIN"]
#  
# Dependencies for vector_retrieval and traditional rag
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

def vector_retrieval(rag_method: str, query: str, uploaded_file_name: str, file_content: bytes, fine_tuning: Optional[FineTuning] = None):
    
    ## Defining the finetuned variables
    recieved_llm = fine_tuning.llm if fine_tuning and fine_tuning.llm else "llama-3.1-70b-versatile"
    if recieved_llm == "gpt-3.5-turbo":
        llm = ChatOpenAI(model=recieved_llm) ##recieved_llm
    else:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

    ## Prompt Template
    prompt_template = """
    You are a helpful assistant. Given the following context and question, provide a detailed and relevant answer.

    Context: {context}

    Question: {question}

    Answer:
    """

    # Handle the file content based on its type (text or binary)
    if uploaded_file_name.endswith('.txt'):
        file_content_str = uploaded_file_content.decode('utf-8')  # Decode as UTF-8 for text files
    
    ############ RAG for PDF
    ## If file is pdf
    elif uploaded_file_name.endswith('.pdf'):
        # Write the uploaded file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_content)
            temp_pdf_path = temp_pdf.name
        #file_content_str = " ".join([page.page_content for page in pages])

        loader = PyPDFLoader(temp_pdf_path)  # Initialize the PDF loader
        pages = loader.load_and_split()  # Load and split PDF content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(pages)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        def format_docs(pages):
            return "\n\n".join(i.page_content for i in pages)
        
        rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

        file_content_str = rag_chain.invoke(query)
    
    else:
        file_content_str = "[Unsupported file type]"  # Handle unsupported file types
        
    return file_content_str

def multi_modal_rag(rag_method: str, query: str, fine_tuning: Optional[FineTuning] = None):
    temp_var = rag_method + " " + query
    if fine_tuning:
        fine_tuning_str = ", ".join(f"{k}={v}" for k, v in fine_tuning.dict(exclude_none=True).items())
        temp_var += f" (Fine-tuning: {fine_tuning_str})"
    return "Multi-modal RAG: " + temp_var

def agentic_rag(rag_method: str, query: str, fine_tuning: Optional[FineTuning] = None):
    temp_var = rag_method + " " + query
    if fine_tuning:
        fine_tuning_str = ", ".join(f"{k}={v}" for k, v in fine_tuning.dict(exclude_none=True).items())
        temp_var += f" (Fine-tuning: {fine_tuning_str})"
    return "Agentic RAG: " + temp_var

def graph_rag(rag_method: str, query: str, fine_tuning: Optional[FineTuning] = None):
    temp_var = rag_method + " " + query
    if fine_tuning:
        fine_tuning_str = ", ".join(f"{k}={v}" for k, v in fine_tuning.dict(exclude_none=True).items())
        temp_var += f" (Fine-tuning: {fine_tuning_str})"
    return "Graph RAG: " + temp_var

#######################################################################################################

