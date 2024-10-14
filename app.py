from fastapi import FastAPI, File, HTTPException, UploadFile
# from dotenv import find_dotenv, load_dotenv
from get_embedding import get_embedding

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

from pathlib import Path
from typing import List

import psutil
import shutil




app = FastAPI()

CHROMA_PATH= "chroma_db"
PROMPT_TEMPLATE = """Answer the question based only on the following context:
    {context}

    Question: {question}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
model = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt_template
#     | model
#     | StrOutputParser()
# )

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding(),
    collection_name="default_collection"
)


# add middlewares here TODO


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/generate")
async def generate_text(query: str):
    response_text = model.invoke(query)
    formatted_response = f"Response:\n{response_text}"
    print(formatted_response)

    return formatted_response


@app.post("/generate-text-with-context")
async def generate_text_with_context(query: str):
    # Search the DB.
    results = vector_store.similarity_search_with_relevance_scores(query, k=3)
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    print(results)

    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_template.format(context=context, question=query)
    print(prompt)

    response_text = model.predict(prompt)
    formatted_response = f"Response:\n{response_text}"
    print(formatted_response)

    return formatted_response


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
        print("after spliting: ", documents)
        await vector_store.aadd_documents(documents)

        return {"message": f"{len(files)} files uploaded and stored successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Remove the temp folder
        shutil.rmtree(TEMPFILE_FOLDER)

@app.get("/system-metrics")
async def system_metrics():
    # def get_memory_info():
    #     return {
    #         "total_memory": psutil.virtual_memory().total / (1024.0 ** 3),
    #         "available_memory": psutil.virtual_memory().available / (1024.0 ** 3),
    #         "used_memory": psutil.virtual_memory().used / (1024.0 ** 3),
    #         "memory_percentage": psutil.virtual_memory().percent
    #     }

    # def get_cpu_info():
    #     return {
    #         "physical_cores": psutil.cpu_count(logical=False),
    #         "total_cores": psutil.cpu_count(logical=True),
    #         "processor_speed": psutil.cpu_freq().current,
    #         "cpu_usage_per_core": dict(enumerate(psutil.cpu_percent(percpu=True, interval=1))),
    #         "total_cpu_usage": psutil.cpu_percent(interval=1)
    #     }

    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_info.percent
    }