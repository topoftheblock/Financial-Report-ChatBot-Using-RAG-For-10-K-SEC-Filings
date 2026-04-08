# Financial Report ChatBot Using RAG

A sophisticated **Retrieval-Augmented Generation (RAG)** system for financial analysis of SEC 10-K reports. By combining Large Language Models, LangChain, and ChromaDB, the system lets users query dense financial filings to extract both qualitative insights (risk factors, business summaries) and quantitative data (revenue tables, operating margins) with high accuracy.

---

## System Architecture

The pipeline is divided into modular components that take data from raw source to an interactive conversational agent.

### 1. Data Ingestion

`src/sec_10k_scraper.py` interfaces with the **SEC EDGAR** database to download raw HTML 10-K filings for specified company tickers and years, storing them in `data/raw/`.

### 2. Parsing & Preprocessing

`src/ingestion/parser.py` converts messy HTML into clean Markdown:

- **Header detection** — identifies SEC-specific section headers (e.g., `PART I`, `Item 1. Business`) and maps them to Markdown headings.
- **Table extraction** — uses BeautifulSoup and Pandas to isolate financial tables, strip formatting artifacts, merge spanning multi-row headers, drop redundant columns, and output clean Markdown tables.

### 3. Chunking & Vectorization

`src/ingestion/chunker.py` processes the parsed Markdown files:

- **Context-aware splitting** — uses LangChain's `MarkdownHeaderTextSplitter` to keep tables and paragraphs tethered to their section headers, never splitting them arbitrarily.
- **Metadata tagging** — each chunk is tagged with `ticker`, `year`, `document_type`, and `section` before being embedded and stored in a local ChromaDB instance (`chroma_financial_db/`).

### 4. Agent & Tooling

The reasoning engine is a streamlined **LangChain Tool-Calling Agent** (`src/agent/generator.py` and `tools.py`) that autonomously selects from specialized tools to answer each user query:

| Tool | Purpose |
|---|---|
| `semantic_financial_search` | Performs semantic similarity search across texts and Markdown tables. Dynamically applies ChromaDB metadata filters (Ticker, Year) to prevent hallucinating or mixing numbers across different filings. |
| `python_calculator` | Python AST REPL code execution sandbox. Used by the agent to accurately compute percentages, YoY growth, and differences based strictly on the retrieved numbers. |

---

## How It Works

### The Core Components

#### A. The Reasoning Engine (The "Brain")

At the center is an LLM (like GPT-4o) specifically fine-tuned for function calling. Instead of just generating text, the LLM is given a JSON schema of available tools. When a user asks a question, the LLM pauses, looks at its tools, and decides: *"Do I know the answer, or do I need to trigger a tool to find out?"*

#### B. The Tools (The "Hands")

The agent is completely blind to your financial data until it uses its tools. Two highly specialized tools are provided:

- **`semantic_financial_search`** — This is the RAG pipeline. It takes the agent's semantic query, converts it into a vector, and searches the local ChromaDB. Crucially, it uses metadata filtering (`ticker` and `year`) to ensure the agent doesn't hallucinate or mix up Apple's 2025 data with Boeing's 2024 data.
- **`python_calculator`** — Financial analysis requires exact math. LLMs are notoriously unreliable at arithmetic. By providing a Python AST REPL, the agent can write actual Python code (e.g., `(15000 - 12000) / 12000`) to compute YoY growth or profit margins with 100% accuracy.

#### C. Short-Term Memory (Context)

The agent maintains an active state of the conversation using LangChain's `HumanMessage` and `AIMessage` objects. If a user asks *"What was Apple's revenue?"* and follows up with *"Calculate the growth from last year,"* the agent uses its memory to know that the second question still refers to Apple.

### The Execution Loop

When a user submits a prompt, the agent enters a reasoning loop and will not return a final answer until it is satisfied it has all the necessary data.

**Example prompt:** *"What were Apple's top risk factors in 2025 and how did their revenue grow?"*

1. **Plan** — The agent reads the prompt and identifies it needs two things: qualitative risk data and quantitative revenue data.
2. **Action 1 (Retrieval)** — Calls `semantic_financial_search(query="top risk factors", company_ticker="AAPL", year=2025)`.
3. **Observation 1** — The database returns chunks of text detailing Apple's legal and supply chain risks.
4. **Action 2 (Retrieval)** — Calls `semantic_financial_search(query="revenue figures", company_ticker="AAPL", year=2025)`.
5. **Observation 2** — The database returns a Markdown table showing $390B for 2025 and $383B for 2024.
6. **Action 3 (Calculation)** — The agent calls `python_calculator(query="((390 - 383) / 383) * 100")`.
7. **Observation 3** — The calculator returns `1.82`.
8. **Final Synthesis** — The agent stops calling tools and synthesizes all observations into a polished, natural-language response complete with inline citations as mandated in `prompt.py`.

### Why This Architecture Is Ideal for Financial RAG

- **Zero hallucination arithmetic** — By forcing the LLM to use `python_calculator` for math, the risk of the model guessing numbers is eliminated entirely.
- **Deterministic filtering** — Because `semantic_financial_search` accepts `company_ticker` and `year` as explicit arguments, the LLM natively pre-filters the vector database. A search for Apple will structurally ignore all other filings, ensuring high-fidelity retrieval.
- **Full transparency** — The Streamlit UI exposes the exact internal `Thought → Action → Observation` loop, making it straightforward to debug any incorrect answers.
---

## Demo

Here is a look at the Financial RAG Agent in action:

### 1. Conversational Interface & Contextual Citations
![Chat Interface](demo1.png)
*The agent synthesizes complex financial queries into clear, professional bullet points. It strictly cites its sources directly from the retrieved SEC filings using inline metadata (Ticker, Year, Section).*

### 2. Transparent Agent Reasoning
![Agent Thoughts](demo2.png)
*Users can expand the "View Agent Thought Process" tab to see exactly how the LangChain agent is thinking. It displays the internal reasoning, the tools called (e.g., `semantic_financial_search`, `python_calculator`), and the raw data retrieved from the vector database before it generates the final answer.*

### 3. Sidebar Controls & Architecture
![Sidebar Architecture](demo3.png)
*The sidebar provides a quick overview of the agent's capabilities, controls to clear the chat history, and an interactive toggle to view the system's underlying LangChain Tool-Calling architecture.*

---
## Directory Structure

```text
Financial-Report-ChatBot-Using-RAG/
│
├── app/                            # Frontend interface
│   ├── main.py                     # Streamlit application entry point
│   └── components.py               # UI components and thought-process renderers
│
├── chroma_financial_db/            # Local ChromaDB vector store
│
├── data/
│   ├── raw/                        # Raw downloaded SEC HTML filings
│   └── processed/                  # Cleaned and parsed Markdown (.txt) files
│
├── src/
│   ├── agent/
│   │   ├── generator.py            # LangChain Agent initialization and execution loop
│   │   ├── tools.py                # Database retrieval and Python calculator tools
│   │   ├── prompt.py               # Core system instructions and formatting rules
│   │   └── config.py               # Agent configuration and token limits
│   │
│   ├── ingestion/
│   │   ├── parser.py               # HTML → Markdown parsing logic
│   │   └── chunker.py              # Text splitting and ChromaDB embedding script
│   │
│   ├── tests/                      # Unit and integration tests
│   └── sec_10k_scraper.py          # SEC EDGAR downloader script
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- An active OpenAI API key

### Steps

**1. Clone the repository**

```bash
git clone <repository-url>
cd Financial-Report-ChatBot-Using-RAG
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the root directory:

```plaintext
OPENAI_API_KEY="your_openai_api_key_here"
```

---

## Usage

Run the following steps in sequence to operate the full pipeline from scratch.

**Step 1 — Scrape financial data**

```bash
python src/sec_10k_scraper.py
```

**Step 2 — Parse HTML to Markdown**

```bash
python src/ingestion/parser.py
```

**Step 3 — Chunk and populate the database**

```bash
python -m src.ingestion.chunker
```

**Step 4 — Launch the application**

```bash
python -m streamlit run app/main.py
```

---

## Testing

Run individual component tests as modules from the root directory to automatically resolve Python path issues.

```bash
# Test the HTML parser
python -m pytest src/tests/test_parser.py

# Test the Markdown chunker
python -m pytest src/tests/test_chunker.py

# Test database retrieval performance
python -m pytest src/tests/test_retrieval_performance.py
```