from tenacity import retry, wait_fixed, stop_after_attempt
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def download_blob_content(blob_service_client, container_name, blob_name, local_folder):
    # Get a blob client
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # Get the content of the blob
    local_path = os.path.join(local_folder, blob_name)
    directory = os.path.dirname(local_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Download the blob to the local path
    with open(local_path, "wb") as file:
        file.write(blob_client.download_blob().readall())
    blob_data = blob_client.download_blob()
    return local_path

def parse_pdf(local_path):
    loader = PyPDFLoader(local_path)
    pages = loader.load_and_split()
    return pages

def split_text(pages, chunk_size=2000, chunk_overlap=500, length_function=len):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

@retry(wait=wait_fixed(35), stop=stop_after_attempt(3))
def generate_chunk_embeddings(openai_client, chunks, model="text-embedding-ada-002"): # model = "deployment_name"
    chunks_text = [chunk.page_content for chunk in chunks]
    embeddings = openai_client.embeddings.create(input = chunks_text, model=model)
    return [result.embedding for result in embeddings.data]

def upload_embeddings_to_search(index_client, index_name, embeddings, chunks, company, file_name):
    docs = [
    {
        "parent_id": "0",
        "chunk_id": f"{chunk.metadata.get('source').split('/')[-1].split('.')[0]}_{i}",
        "chunk": chunk.page_content,
        "company": company,
        "title": chunk.metadata.get('source').split('/')[-1],
        "vector": embeddings[i]
    }
    for i, chunk in enumerate(chunks)
    ]
    search_client = index_client.get_search_client(index_name)
    try:
        search_client.upload_documents(docs)
        nb_chunks = len(chunks)
        print(f"Uploaded {nb_chunks} chunks and embeddings for {file_name}")
    
    except Exception as e:
        print(f"Error uploading documents: {e}")