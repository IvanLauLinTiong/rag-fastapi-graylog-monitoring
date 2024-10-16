from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from dotenv import find_dotenv, load_dotenv
from get_embedding import get_embedding

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

from logger import logger
from pathlib import Path
from typing import List
from utils import get_system_metrics
import os
import shutil
import time



load_dotenv(find_dotenv())


# async def send_metrics_to_graylog_periodically(interval: int):
#     while True:
#         metrics = get_system_metrics()
#         logger.info("System Metrics", extra=metrics)
#         time.sleep(interval)

# background_tasks = BackgroundTasks()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     background_tasks.add_task(send_metrics_to_graylog_periodically, 60)
#     await background_tasks()
#     yield


# app = FastAPI(lifespan=lifespan)

app = FastAPI()

CHROMA_PATH= "chroma_db"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# # model = HuggingFacePipeline.from_model_id(
# #     model_id=MODEL_ID,
# #     task="text-generation",
# #     pipeline_kwargs={"max_new_tokens": 256},
# # )

# model = ""

model = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding(),
    collection_name="default_collection"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# --- Middlewares ---
@app.middleware("http")
async def log_requests(request: Request, call_next):

    logger.info(f"Incoming request from IP {request.client.host}: {request.method} {request.url}")

    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    print(f"Response status: {response.status_code}\nAPI response time: {elapsed}")

    logger.info(f"Completed request in {elapsed:.2f} seconds, status: {response.status_code}")

    response.headers["X-Response-Time"] = str(elapsed)

    return response

# ----- APIs ---
@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/generate")
async def generate_text(query: str):
    template = """Question: {question}

    Answer: """
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    response_text = chain.invoke({"question": query})

    formatted_response = f"Response:\n{response_text}"
    print(formatted_response)

    return formatted_response


@app.post("/generate-text-with-context")
async def generate_text_with_context(query: str):
    template = """You are a Q&A assistant. Your goal is to answer the question based on the provided context and rules below:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the answer for the context provided".
    2. If you find the answer, write the answer in a concise way.

    Context: {context}

    Question: {question}

    Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response_text = chain.invoke(query)
    print(response_text)

    return response_text


@app.post("/upload")
async def create_database(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            # Save each uploaded file temporarily
            TEMPFILE_FOLDER = "./temp"
            Path("./temp").mkdir(exist_ok=True)
            temp_file_path = f"{TEMPFILE_FOLDER}/{file.filename}"

            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())


        # Add uploaded docs to chromadb
        loader = PyPDFDirectoryLoader(TEMPFILE_FOLDER)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24)
        documents = text_splitter.split_documents(documents)
        await vector_store.aadd_documents(documents)

        return {"message": f"{len(files)} files uploaded and stored successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Remove the temp folder
        shutil.rmtree(TEMPFILE_FOLDER)

@app.get("/system-metrics")
async def system_metrics():
    metrics = get_system_metrics()
    logger.info("System Metrics", extra=metrics)
    return metrics