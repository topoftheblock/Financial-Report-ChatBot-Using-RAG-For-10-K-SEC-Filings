import os
from src.ingestion.parser import parse_10k_html

def run_parser_test():
    # Set up test directories
    TEST_DIR = "test_data"
    RAW_DIR = os.path.join(TEST_DIR, "raw")
    PROCESSED_DIR = os.path.join(TEST_DIR, "processed")

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Create a dummy SEC 10-K HTML file with headers and a financial table
    mock_html = """
    <html>
    <body>
        <div><span style="font-weight: bold">PART I</span></div>
        <div><span style="font-weight: 700">Item 1. Business</span></div>
        <p>This is the business description of the test company.</p>
        
        <div><span style="font-weight: bold">Item 7. Management's Discussion</span></div>
        <p>Financial overview:</p>
        <table>
            <tr>
                <td></td>
                <td>2024</td>
                <td>2023</td>
            </tr>
            <tr>
                <td>Revenue</td>
                <td>$ 10,000</td>
                <td>$ 8,500</td>
            </tr>
            <tr>
                <td>Cost of Goods Sold</td>
                <td>$ 4,000</td>
                <td>$ 3,500</td>
            </tr>
        </table>
    </body>
    </html>
    """
    test_file_path = os.path.join(RAW_DIR, "test_10k.html")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(mock_html)
    print(f"Created test HTML at {test_file_path}")

    # Run the parser
    print("\nRunning parser...")
    parse_10k_html(test_file_path, PROCESSED_DIR)

    # Read the output
    output_file = os.path.join(PROCESSED_DIR, "10-K.txt")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            parsed_markdown = f.read()

        print("\n=== PARSED MARKDOWN OUTPUT ===")
        print(parsed_markdown)
    else:
        print(f"\nError: Could not find output file at {output_file}")

if __name__ == "__main__":
    run_parser_test()