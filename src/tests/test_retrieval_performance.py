import time
from src.agent.tools import perform_metadata_search

def test_retrieval(query: str, ticker: str, year: int, doc_type: str = "10-K"):
    filters = {
        "ticker": ticker.upper(),
        "year": year,
        "document_type": doc_type.upper()
    }
    
    print(f"Query: '{query}' | Filters: {filters}")
    
    start_time = time.time()
    
    # perform_metadata_search requests 30 results from ChromaDB then post-filters to 5
    result_text = perform_metadata_search(query=query, filters=filters, n_results=5)
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Calculate how many chunks were returned based on the separator used in tools.py
    if "No financial documents found" in result_text or "Error" in result_text:
        num_chunks = 0
    else:
        num_chunks = len(result_text.split("\n\n---\n\n"))
        
    print(f"Latency: {latency:.4f} seconds")
    print(f"Chunks Retrieved: {num_chunks}")
    print("-" * 50)
    
    return result_text

def run_performance_tests():
    print("Ensure your ChromaDB path is populated with data before running these tests.\n")
    
    # Scenario 1: Basic text retrieval
    print("SCENARIO 1: Qualitative Text Search")
    res1 = test_retrieval(
        query="What are the primary risk factors regarding supply chain?", 
        ticker="AAPL", 
        year=2024
    )

    # Scenario 2: Table/Metric retrieval
    print("\nSCENARIO 2: Quantitative Table Search")
    res2 = test_retrieval(
        query="Total Net Sales Revenue table", 
        ticker="AAPL", 
        year=2024
    )

    # Scenario 3: Checking metadata filter accuracy (Querying a year that likely doesn't exist)
    print("\nSCENARIO 3: Empty Filter Test")
    res3 = test_retrieval(
        query="Revenue", 
        ticker="AAPL", 
        year=2099 
    )
    
    if "No financial documents found" not in res2 and "Error" not in res2:
        print("\n=== PREVIEW OF QUANTITATIVE RETRIEVAL (SCENARIO 2) ===")
        # Print a snippet of the table search to verify markdown formatting survived
        print(res2[:1000] + "\n...[truncated]...")

if __name__ == "__main__":
    run_performance_tests()