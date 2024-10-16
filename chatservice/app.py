from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from dotenv import find_dotenv, load_dotenv
from get_embedding import get_embedding

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
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
app = FastAPI()

CHROMA_PATH= "chroma_db"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# # model = HuggingFacePipeline.from_model_id(
# #     model_id=MODEL_ID,
# #     task="text-generation",
# #     pipeline_kwargs={"max_new_tokens": 256},
# # )


model = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    max_new_tokens=256,
    temperature=0.001,
    stop_sequences=["\n\n"],
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding(),
    collection_name="default_collection"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# @app.middleware("http")
async def log_requests(request: Request, call_next):

    logger.info(f"Incoming request from IP {request.client.host}: {request.method} {request.url}")

    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    print(f"Response status: {response.status_code}\nAPI response time: {elapsed}")

    logger.info(f"Completed request in {elapsed:.2f} seconds, status: {response.status_code}")

    response.headers["X-Response-Time"] = str(elapsed)

    return response


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/generate")
async def generate_text(query: str):
    template = """You are a Q&A assistant. Your goal is to answer the question below.
    If you don't know the answer, say you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}

    Answer:
"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    chain = prompt | model

    response_text = chain.invoke({"question": query})

    return response_text


@app.post("/generate-text-with-context")
async def generate_text_with_context(query: str):
    template = """You are a Q&A assistant. Your goal is to answer the question based on the provided context.
    If you don't know the answer, say you don't know. Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:
"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])


    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        # return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
)

    result = chain.invoke({"query": query})
    response_text = result["result"]
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