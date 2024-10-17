
# FastAPI-RAG-Graylog

A FastAPI-based service implementing a Retrieval-Augmented Generation (RAG) pipeline for text generation, integrated with Graylog for API monitoring. The service allows users to upload files, store them in a Chroma vector store, and generate context-aware text using a pre-trained model endpoint ([microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)) from HuggingFace.

## Features
- Text Generation: Generate concise answers based on provided queries or retrieved document context.
- File Upload: Upload multiple PDF files, split them into chunks, and store them in a Chroma vector store for retrieval.
- Monitoring: Logs incoming requests, system metrics, and API performance (response time) to Graylog.

## How it works


#### API Endpoints

```
POST /generate

Request:
- query: the question or prompt for text generation.

Response:
Generated text based on the query (without context awareness).
```
<br />

```
POST /generate-text-with-context

Request:
- query: the question or prompt for text generation.

Response:
A context-aware answer based on the uploaded documents stored in the vector database.
```
<br />


```
POST /upload

Request:
- files: list of PDF files to be uploaded, processed and stored in vector store.

Response:
Number of files uploaded and stored successfully.
```
<br />

```
GET /system-metrics

Response:
Returns CPU, memory, and disk usage metrics and logs to graylog server.
```
<br />

```
GET /health

Response:
Returns the health status of the service.
```




## Setup


## Caveats