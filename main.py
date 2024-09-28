import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

######### Loading Dotenv
import os
from dotenv import load_dotenv
load_dotenv()
#########
app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        ssl_keyfile="/etc/letsencrypt/live/64-227-185-75.nip.io/privkey.pem", 
        ssl_certfile="/etc/letsencrypt/live/64-227-185-75.nip.io/fullchain.pem"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN")],
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

## Environment varibale initialization
#NEO4J_URI = os.getenv("NEO4J_URI")
#NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
#NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
#NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
groq_api_key = os.getenv("groq_api_key")
OPENAI_API_KEY2 = os.getenv('OPENAI_API_KEY')
#LANGCHAIN_TRACING_V2 = "true"
#LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
#LANGCHAIN_PROJECT="RAG_Playground"

#### INITIALIZE FT METHODS SENT BY USER
## no change for deployment


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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []  # Chat history is used by qa_prompt and history_aware_retriever
chat_history.clear()
def vector_retrieval(rag_method: str, query: str, uploaded_file_name: str, file_content: bytes, fine_tuning: Optional[FineTuning] = None):
    
    ## Defining the finetuned variables
    recieved_llm = fine_tuning.llm if fine_tuning and fine_tuning.llm else "llama-3.1-70b-versatile"
    if recieved_llm == "gpt-3.5-turbo":
        llm = ChatOpenAI(model=recieved_llm) ##recieved_llm
    elif recieved_llm == "gpt-4o-mini":
        llm = ChatOpenAI(model=recieved_llm)
    elif recieved_llm == "gpt-4o":
        llm = ChatOpenAI(model=recieved_llm)
    elif recieved_llm == "llama-3.2-90b-text-preview":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    elif recieved_llm == "llama-3.2-11b-text-preview":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    elif recieved_llm == "llama-3.2-1b-preview":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    elif recieved_llm == "llama-3.1-8b-instant":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    elif recieved_llm == "mixtral-8x7b-32768":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    elif recieved_llm == "gemma-7b-it":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    elif recieved_llm == "gemma2-9b-it":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=recieved_llm)
    else:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
    
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
        
        contextualize_q_system_prompt = ( 
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),  # System prompt that instructs LLM to restructure a prompt with context if the latest query refers to past
                MessagesPlaceholder("chat_history"),  # Chat_history is a list of messages
                ("human", "{input}"),  # Human prompt is the latest query
            ]
        )

        # Assuming you have correctly initialized `llm` and `retriever` objects
        history_aware_retriever = create_history_aware_retriever(
            llm=llm, retriever=retriever, prompt=contextualize_q_prompt
        )

        system_prompt1 = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt1),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Ensure `llm` is correctly initialized
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # New rag_chain using history_aware_retriever
        
        ai_message = rag_chain.invoke({"input": query, "chat_history": chat_history})
        chat_history.extend(
                            [
                                HumanMessage(content=query),
                                AIMessage(content=ai_message["answer"]),
                            ]
                        )
        n = len(chat_history)
        print("Answer", ai_message["answer"])
        print("Chat Hisotry", chat_history)
    
    else:
        ai_message = "[Unsupported file type]"  # Handle unsupported file types
        
    return ai_message["answer"]

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

