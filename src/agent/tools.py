import os
import chromadb
import pandas as pd
import io
import re
from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "chroma_financial_db")
COLLECTION_NAME = "financial_statements"

def perform_metadata_search(query: str, filters: dict, n_results: int = 5) -> str:
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        chroma_filter = {"ticker": filters["ticker"]} if "ticker" in filters else None
        results = collection.query(query_texts=[query], n_results=30, where=chroma_filter)
        
        if not results['documents'] or not results['documents'][0]:
            return "No documents found."
            
        valid_chunks = []
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            chunk = results['documents'][0][i]
            
            if all(meta.get(k) == v for k, v in filters.items()):
                valid_chunks.append(chunk)
                if len(valid_chunks) == n_results: break
                    
        return "\n\n---\n\n".join(valid_chunks) if valid_chunks else "No exact matches."
    except Exception as e:
        return f"Database error: {str(e)}"

@tool
def fetch_sec_section(company_ticker: str, year: int, section_name: str) -> str:
    """Fetches an exact SEC Markdown section by name (e.g., 'Item 3' for Legal, 'Item 7' for MD&A)."""
    filters = {"ticker": company_ticker.upper(), "year": year, "Item": section_name}
    return perform_metadata_search(query="financials", filters=filters, n_results=3)

@tool
def search_unstructured_text(query: str, company_ticker: str, year: int) -> str:
    """Performs semantic search over the 10-K for qualitative questions (e.g., legal, risks, business)."""
    filters = {"ticker": company_ticker.upper(), "year": year}
    return perform_metadata_search(query=query, filters=filters, n_results=5)

@tool
def extract_and_query_markdown_table(markdown_table: str, query_code: str) -> str:
    """Parses a Markdown table into Pandas and runs query_code."""
    try:
        cleaned = re.sub(r'(^\|)|(\|$)', '', markdown_table, flags=re.MULTILINE)
        lines = [line for line in cleaned.split('\n') if not re.match(r'^[\s\-|:]+$', line)]
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep='|')
        df.columns = df.columns.str.strip()
        
        local_env = {"df": df, "pd": pd}
        exec(f"result = {query_code}", local_env)
        return f"Table Query Result: {local_env.get('result', 'Success')}"
    except Exception as e:
        return f"Table error: {e}"

python_calculator = PythonAstREPLTool(
    name="python_calculator",
    description="Python shell. Use this to execute math."
)

def get_researcher_tools(): 
    return [fetch_sec_section, search_unstructured_text]

def get_quant_tools(): 
    return [extract_and_query_markdown_table, python_calculator]
