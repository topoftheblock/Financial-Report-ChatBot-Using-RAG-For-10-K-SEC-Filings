import os
from src.ingestion.chunker import chunk_markdown_file

def run_chunker_test():
    # Simulate the markdown output from the parser
    test_md_path = "test_parsed_10k.md"
    mock_markdown = """
# PART I

## Item 1. Business
Apple Inc. designs, manufactures, and markets smartphones, personal computers, and wearables.

## Item 7. Management's Discussion
The following table shows our net sales by category:

| Category | 2024 | 2023 |
|---|---|---|
| iPhone | $ 200,000 | $ 190,000 |
| Mac | $ 40,000 | $ 38,000 |
| iPad | $ 25,000 | $ 24,000 |

### Risk Factors
Macroeconomic conditions could affect our margins.
"""

    with open(test_md_path, "w", encoding="utf-8") as f:
        f.write(mock_markdown.strip())
        
    print(f"Created test Markdown file at {test_md_path}\n")

    # Create base metadata simulating a specific company filing
    base_metadata = {
        "company": "AAPL",
        "ticker": "AAPL",
        "document_type": "10-K",
        "year": 2024
    }

    # Run the chunker
    print("Running chunker...")
    chunks = chunk_markdown_file(test_md_path, base_metadata)

    print(f"Total chunks created: {len(chunks)}\n")

    # Display how headers become metadata and tables stay intact
    for i, chunk in enumerate(chunks):
        print(f"--- CHUNK {i+1} ---")
        print(f"Metadata Inherited: {chunk.metadata}")
        print(f"Content Length: {len(chunk.page_content)} characters")
        print(f"Content Snippet:\n{chunk.page_content}\n")
        
    # Cleanup
    if os.path.exists(test_md_path):
        os.remove(test_md_path)

if __name__ == "__main__":
    run_chunker_test()