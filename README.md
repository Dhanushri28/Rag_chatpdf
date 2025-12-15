🌐 Live Demo

🚀 Try the application live:
🔗 https://lite-chatpdf.streamlit.app/

Upload a PDF and ask questions in natural language to receive accurate, document-grounded answers in real time.


Building a Question Answering System with PDFs, Embeddings, and FAISS

This document outlines the process of building a question answering system that can extract information from PDF documents, convert the text into numerical representations (embeddings), store these embeddings in a FAISS vector store for efficient similarity search, and finally, use a Large Language Model (LLM) to generate answers based on the retrieved information. The process involves several key steps, from uploading the PDF to providing the user with a coherent answer.



1. PDF Upload



The initial step involves uploading the PDF document that will serve as the knowledge base for the question answering system. This can be achieved through a web interface, API endpoint, or a local file system interface. The system should be able to handle various PDF formats and sizes, with appropriate error handling for corrupted or unsupported files.



2. Text Extraction



Once the PDF is uploaded, the next step is to extract the text content from the document. This can be accomplished using libraries like PyPDF2, PDFMiner, or OCR (Optical Character Recognition) tools for scanned documents.







PyPDF2: A pure-Python PDF library capable of extracting text, metadata, and other information from PDF files. It's relatively simple to use but may struggle with complex layouts or scanned documents.



```python

import PyPDF2



def extract_text_from_pdf(pdf_path):

    text = ""

    with open(pdf_path, 'rb') as file:

        reader = PyPDF2.PdfReader(file)

        for page_num in range(len(reader.pages)):

            page = reader.pages[page_num]

            text += page.extract_text()

    return text

```







PDFMiner: Another Python library for extracting text from PDFs. It's more robust than PyPDF2 and can handle more complex layouts. PDFMiner.six is a maintained fork of the original PDFMiner.



```python

from pdfminer.high_level import extract_text



def extract_text_from_pdf_miner(pdf_path):

    text = extract_text(pdf_path)

    return text

```









import pytesseract

from PIL import Image



def extract_text_from_image(image_path):

    text = pytesseract.image_to_string(Image.open(image_path))

    return text

```



The extracted text may contain noise, such as headers, footers, and formatting characters. Preprocessing steps like removing unnecessary whitespace, special characters, and potentially applying stemming or lemmatization can improve the quality of the text.



3. Text Chunking



Large documents are often broken down into smaller, more manageable chunks. This is crucial for several reasons:







LLM Input Limits: LLMs have a limited input token size. Chunking ensures that the input to the LLM stays within these limits.



Relevance: Smaller chunks are more likely to contain relevant information for a specific query.



Efficiency: Processing smaller chunks is generally faster and more efficient.



Common chunking strategies include:







Fixed-Size Chunking: Dividing the text into chunks of a fixed number of characters or words.



Semantic Chunking: Splitting the text based on sentence boundaries, paragraph breaks, or other semantic cues. Libraries like nltk can be used for sentence tokenization.



Recursive Chunking:  Breaking down the text recursively until each chunk meets a certain size criterion. This can help preserve context across chunks.



import nltk

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks




4. Embeddings Generation



Embeddings are numerical representations of text that capture its semantic meaning. These embeddings allow the system to compare the similarity between different text chunks and the user's query.







Word Embeddings:  Models like Word2Vec, GloVe, and FastText can be used to generate embeddings for individual words. However, they may not capture the context of the entire chunk effectively.



Sentence Embeddings: Models like Sentence Transformers are specifically designed to generate embeddings for sentences and paragraphs. They provide a more accurate representation of the semantic meaning of the text chunks.



from sentence_transformers import SentenceTransformer

def generate_embeddings(text_chunks, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks)
    return embeddings




5. FAISS Vector Store



FAISS (Facebook AI Similarity Search) is a library for efficient similarity search in high-dimensional spaces. It allows the system to quickly find the text chunks that are most similar to the user's query.







Indexing: The embeddings generated in the previous step are indexed in the FAISS vector store. This involves creating a data structure that allows for fast similarity search.



Similarity Search: When a user submits a query, the query is also converted into an embedding. The FAISS index is then used to find the text chunks with the highest similarity scores to the query embedding.



import faiss
import numpy as np

def create_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices




6. Similarity Search



The user's query is embedded using the same model used for the text chunks. This query embedding is then used to search the FAISS index for the most similar text chunks. The number of chunks retrieved (top-k) can be adjusted based on the desired level of accuracy and the complexity of the query.



7. LLM Response Generation



The retrieved text chunks are then fed into a Large Language Model (LLM) along with the user's query. The LLM uses this information to generate a coherent and informative answer.







Prompt Engineering: The prompt given to the LLM is crucial for generating a good answer. The prompt should include the user's query, the retrieved text chunks, and instructions on how to generate the answer.



LLM Selection:  Models like GPT-3, GPT-4, or open-source alternatives like Llama 2 can be used. The choice of model depends on the desired level of accuracy, cost, and availability.



from transformers import pipeline

def generate_answer(query, context, model_name='google/flan-t5-base'):
    qa_pipeline = pipeline("question-answering", model=model_name)
    result = qa_pipeline(question=query, context=context)
    return result['answer']




8. User Answer



The final step is to present the generated answer to the user. The answer should be clear, concise, and relevant to the user's query. The system may also provide citations or references to the original PDF document to allow the user to verify the information.



This entire process enables a powerful question-answering system capable of extracting, understanding, and responding to queries based on the content of PDF documents. By leveraging embeddings and FAISS, the system can efficiently search for relevant information, while LLMs provide the ability to generate human-quality answers.
