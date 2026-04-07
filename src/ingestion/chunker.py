import os
import chromadb
import os
import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def chunk_markdown_file(file_path: str, base_metadata: dict) -> list:
    with open(file_path, "r", encoding="utf-8") as file:
        markdown_document = file.read()

    # Pass 1: Split by logical headers
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Pass 2: Split by size to ensure embeddability (adjust size/overlap as needed)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6_000, 
        chunk_overlap=400,
        separators=["\n\n", "\n", " ", ""] # Will try not to break tables if possible
    )
    final_splits = text_splitter.split_documents(md_header_splits)

    # Append base metadata to all final chunks
    for split in final_splits:
        split.metadata.update(base_metadata)
        
    return final_splits

def store_vectors_in_chroma(chunks: list, db_path: str, collection_name: str, id_prefix: str) -> None:
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{id_prefix}_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully stored {len(documents)} chunks into '{collection_name}' collection at '{db_path}'.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    DB_PATH = os.path.join(BASE_DIR, "chroma_financial_db")
    COLLECTION_NAME = "financial_statements"

    if not os.path.exists(PROCESSED_DIR):
        print(f"Could not find processed data directory at: {PROCESSED_DIR}")
        exit()

    companies = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]

    for company in companies:
        company_path = os.path.join(PROCESSED_DIR, company)
        years = [y for y in os.listdir(company_path) if os.path.isdir(os.path.join(company_path, y))]
        
        for year in years:
            target_file = os.path.join(company_path, year, "10-K.txt")
            
            if os.path.exists(target_file):
                print(f"Chunking and embedding: {company} - {year}")
                
                company_metadata = {
                    "company": company,
                    "ticker": company,
                    "document_type": "10-K",
                    "year": int(year)
                }
                
                document_chunks = chunk_markdown_file(
                    file_path=target_file, 
                    base_metadata=company_metadata
                )
                
                store_vectors_in_chroma(
                    chunks=document_chunks, 
                    db_path=DB_PATH, 
                    collection_name=COLLECTION_NAME,
                    id_prefix=f"{company}_10K_{year}"
                )

def store_vectors_in_chroma(chunks: list, db_path: str, collection_name: str, id_prefix: str) -> None:
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{id_prefix}_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully stored {len(documents)} chunks into '{collection_name}' collection at '{db_path}'.")

def embed_all_processed_files():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    DB_PATH = os.path.join(BASE_DIR, "chroma_financial_db")
    COLLECTION_NAME = "financial_statements"

    if not os.path.exists(PROCESSED_DIR):
        print(f"Could not find processed data directory at: {PROCESSED_DIR}")
        return

    companies = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]

    for company in companies:
        company_path = os.path.join(PROCESSED_DIR, company)
        years = [y for y in os.listdir(company_path) if os.path.isdir(os.path.join(company_path, y))]
        
        for year in years:
            target_file = os.path.join(company_path, year, "10-K.txt")
            
            if os.path.exists(target_file):
                print(f"Chunking and embedding: {company} - {year}")
                
                company_metadata = {
                    "company": company,
                    "ticker": company,
                    "document_type": "10-K",
                    "year": int(year)
                }
                
                document_chunks = chunk_markdown_file(
                    file_path=target_file, 
                    base_metadata=company_metadata
                )
                
                store_vectors_in_chroma(
                    chunks=document_chunks, 
                    db_path=DB_PATH, 
                    collection_name=COLLECTION_NAME,
                    id_prefix=f"{company}_10K_{year}"
                )

if __name__ == "__main__":
    embed_all_processed_files()