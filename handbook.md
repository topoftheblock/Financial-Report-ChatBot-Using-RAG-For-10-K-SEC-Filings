# Financial Report ChatBot
## Application Handbook
### Enterprise AI Analyst for SEC 10-K Filings — Version 1.0

---

## Table of Contents
1. [Application Overview](#1-application-overview)
2. [Data Acquisition — SEC EDGAR Scraper](#2-data-acquisition--sec-edgar-scraper)
3. [HTML-to-Markdown Parser](#3-html-to-markdown-parser)
4. [Context-Aware Vectorisation](#4-context-aware-vectorisation)
5. [LangChain Agent Architecture](#5-langchain-agent-architecture)
6. [Specialised Tool Suite](#6-specialised-tool-suite)
7. [End-to-End Data Flow](#7-end-to-end-data-flow)
8. [Configuration Reference](#8-configuration-reference)
9. [Known Limitations and Design Decisions](#9-known-limitations-and-design-decisions)

---

## 1. Application Overview

The Financial Report ChatBot is a Streamlit-based enterprise AI analyst that enables users to query dense SEC 10-K filings through a natural-language chat interface. It combines a sophisticated data ingestion pipeline, a specialised HTML parser, a context-aware vector database, and a multi-step LangChain agent to deliver accurate, verifiable financial analysis.

### 1.1 Key Capabilities

| Capability | Description | Technical Basis |
|---|---|---|
| Natural-Language Querying | Ask questions in plain English about any ingested 10-K filing | LangChain Tool-Calling Agent |
| Multi-Company Analysis | Compare financials across companies and fiscal years simultaneously | Sequential agent with memory |
| Precise Arithmetic | Year-over-year growth, margins, and ratios computed exactly | Python sandbox calculator |
| Source Transparency | View every intermediate reasoning step the agent took | Streamlit expander UI |
| Token Safety | Automatic context compression prevents prompt overflow | Dynamic token estimation |
| Duplicate Prevention | Retrieval-key deduplication avoids redundant DB searches | AgentState tracking set |

### 1.2 User Interface

The Streamlit frontend maintains session state for both the chat history and the agent instance.

- **Chat History:** Stored in `st.session_state` and re-rendered each page load.
- **Thought-Process Expander:** A collapsible panel showing tool selections, query parameters, and retrieved data previews.

> **Transparency by Design** — The thought-process expander is a core feature. Financial professionals are expected to audit the agent's reasoning before acting on any result.

---

## 2. Data Acquisition — SEC EDGAR Scraper

A structured ETL pipeline (Extract, Transform, Load) is responsible for downloading and cataloguing raw 10-K filings from the SEC EDGAR public database.

### 2.1 CIK Mapping

The scraper converts each user-supplied ticker symbol into the SEC's internal Central Index Key (CIK). It downloads `company_tickers.json` from EDGAR and performs a case-insensitive lookup, zero-padding the result to the required 10-digit CIK format before any API requests are made.

### 2.2 Filing Validation and Deduplication

Three filters are applied to every filing returned by the EDGAR full-text search API:

- **Form Type:** Only `10-K` submissions are retained; amendments (`10-K/A`) are excluded.
- **Date Range:** A user-defined start and end year restricts the result set.
- **Deduplication:** An in-memory accession-number set prevents the same filing from being downloaded more than once per run.

### 2.3 Metadata Catalogue

Each successfully downloaded filing is appended to `metadata.csv`, which acts as a persistent index of the local corpus.

| Column | Description |
|---|---|
| `ticker` | Stock ticker symbol (e.g. AAPL) |
| `cik` | SEC Central Index Key |
| `company_name` | Official registered company name |
| `filing_date` | Date the 10-K was filed with the SEC |
| `period_of_report` | Fiscal year-end covered by the filing |
| `accession_number` | Unique EDGAR identifier for the filing |
| `local_path` | Relative file path within `data/raw/` |

---

## 3. HTML-to-Markdown Parser

SEC filings are published as dense, poorly structured HTML. The parser (`src/ingestion/parser.py`) converts this into clean, semantically structured Markdown that downstream components can reliably process.

### 3.1 Semantic Header Promotion

The parser scans every block-level element for patterns matching SEC document conventions — `PART I`, `PART II`, `Item 1`, `Item 7`, and similar — and promotes them to the appropriate Markdown heading level (`#` through `###`). This preserves the hierarchical structure of the filing and allows the chunker to split on meaningful boundaries rather than arbitrary character counts.

### 3.2 Financial Table Reconstruction

Raw HTML tables in SEC filings frequently contain merged cells, multi-row headers, and split currency values. The parser applies a sequence of transformations to correct these issues:

- **Multi-row header merging:** Spanning header cells are flattened into a single descriptive label.
- **Empty column removal:** Columns that contain only whitespace or formatting artifacts are dropped entirely.
- **Cell-by-cell currency merging:** Adjacent cells where one contains a currency symbol and the next contains the numeric value are concatenated, turning `$` and `4,521` into `$4,521`.

> **Why This Matters** — Merged currency values directly improve model accuracy. When `$` and `4,521` appear as separate tokens, the LLM may fail to associate them during retrieval or arithmetic. Merging them at parse time eliminates this failure mode entirely.

---

## 4. Context-Aware Vectorisation

The chunker (`src/ingestion/chunker.py`) converts parsed Markdown files into embedded vector chunks stored in a local ChromaDB instance. Two design decisions govern this stage: hierarchical chunking and strict metadata tagging.

### 4.1 Hierarchical Chunking

LangChain's `MarkdownHeaderTextSplitter` is used to split documents along their section boundaries rather than by raw character count. Each chunk inherits the section headers that enclose it, which means a table from Item 7 (Management's Discussion) is always retrieved together with the context that identifies it as such. Tables and paragraphs are never split mid-content.

### 4.2 Strict Metadata Tagging

Before embedding, every chunk is tagged with three metadata fields:

| Field | Purpose |
|---|---|
| `ticker` | Enables the retrieval tools to filter results to a specific company |
| `year` | Enables the retrieval tools to filter results to a specific fiscal year |
| `doc_type` | Distinguishes structured table chunks from unstructured narrative chunks |

> **Critical Safety Guarantee** — Metadata filtering is what prevents the agent from mixing Apple's 2025 revenue with Boeing's 2024 figures. Without it, semantic similarity alone would retrieve topically related but factually incorrect data.

---

## 5. LangChain Agent Architecture

The reasoning engine is a **LangChain Tool-Calling Agent** built on top of a function-calling LLM (GPT-4o). Rather than generating free-form text in a single pass, the agent operates in an iterative **Thought → Action → Observation** loop, invoking specialised tools until it is confident it has gathered all the data needed to answer the user's question accurately.

### 5.1 How the Agent Works

At the centre of the system is a function-calling LLM. When a user submits a query, the LLM is provided with a JSON schema describing every available tool. Instead of answering immediately, it decides: *Do I already know this, or do I need to call a tool to find out?* This decision is repeated after every tool result until the agent reaches a final answer.

The agent maintains conversational context across turns using LangChain's `HumanMessage` and `AIMessage` objects. If a user asks *"What was Apple's revenue?"* and follows up with *"Calculate the growth from last year,"* the agent uses its message history to resolve the implicit reference back to Apple.

### 5.2 State Management

The agent tracks two categories of state across each reasoning session.

**Conversation State**

| Field | Description |
|---|---|
| `conversation_history` | Full list of `HumanMessage` and `AIMessage` objects for the current session |
| `conversation_summary` | Condensed plain-text summary of older turns, used when history exceeds the token threshold |
| `original_query` | The raw user input before any rewriting or decomposition |
| `rewritten_questions` | One or more structured sub-questions derived from the original query |
| `agent_answers` | Aggregated results collected from individual tool calls |

**Execution State**

| Field | Description |
|---|---|
| `tool_call_count` | Running total of tool invocations; capped at `MAX_TOOL_CALLS` to prevent runaway loops |
| `iteration_count` | Number of completed Thought → Action → Observation cycles |
| `context_summary` | Compressed representation of intermediate observations when context grows large |
| `retrieval_keys` | Set of `(query, ticker, year)` tuples already searched, preventing duplicate DB calls |

### 5.3 Query Planning — Decompose and Conquer

Before the agent begins calling tools, a planning step decomposes complex user queries into discrete, answerable sub-questions. For example, *"What were Apple's top risk factors in 2025 and how did their revenue grow?"* is split into two independent retrieval tasks — one qualitative, one quantitative — which are then executed sequentially. The following steps govern the planning phase:

- **`summarize_history`:** Compresses the chat history into a brief summary when it exceeds `HISTORY_THRESHOLD` turns, keeping the active context window lean.
- **`rewrite_query`:** Uses the LLM to reformulate the user's raw input into one or more precise, retrieval-optimised sub-questions.
- **`route_after_rewrite`:** Inspects the rewritten questions and decides whether a single agent pass is sufficient or whether multiple sub-queries must be run.
- **`aggregate_answers`:** Collects the results from all sub-queries and passes them to the final synthesis step.

### 5.4 Execution Loop — ReAct Pattern

Each sub-question is handled by the agent's core **ReAct (Reasoning + Acting)** loop. The orchestrator issues a prompt containing the sub-question, the available tools, and the current context, then enters the following cycle:

1. **Think:** The LLM reasons about what information is still missing.
2. **Act:** The LLM emits a structured tool call (tool name + arguments) rather than a free-text answer.
3. **Observe:** The tool executes and returns a result, which is appended to the context as a `ToolMessage`.
4. **Repeat or Return:** If the LLM decides it has enough data, it emits a final answer. Otherwise the loop continues.

Three safety mechanisms prevent the loop from running indefinitely:

- **`MAX_TOOL_CALLS` (default 10):** Hard cap on the total number of tool invocations per query.
- **`MAX_ITERATIONS` (default 5):** Hard cap on the number of full Thought → Action → Observation cycles.
- **Token compression:** When the accumulated context exceeds `TOKEN_THRESHOLD` characters, intermediate observations are summarised before the next iteration begins.

### 5.5 Final Synthesis

Once the agent exits the execution loop, all collected observations are passed to the LLM for final synthesis. The system prompt in `prompt.py` mandates that every factual claim in the response must be accompanied by an inline citation referencing the specific 10-K filing and section from which the data was retrieved. The synthesised answer and the full Thought → Action → Observation trace are then rendered in the Streamlit UI.

---

## 6. Specialised Tool Suite

The agent is equipped with three tools, each designed to eliminate a distinct class of LLM failure mode. The agent autonomously selects which tool to call based on the nature of each sub-question.

### 6.1 `search_financial_tables`

Performs semantic similarity search restricted to chunks tagged `doc_type = table`. This tool is used whenever the agent needs numerical data: revenue figures, operating margins, segment breakdowns, or any other structured financial information.

Because ChromaDB's native metadata filtering can behave inconsistently when combining semantic and exact-match constraints, the tool uses a two-pass strategy: it first retrieves a broader candidate set (`PRE_FILTER_K` results) using semantic similarity alone, then applies `ticker` and `year` filters in Python before returning the final ranked results. This trades a modest increase in memory usage for deterministic filtering behaviour.

### 6.2 `search_unstructured_text`

Performs semantic similarity search restricted to chunks tagged `doc_type = text`. This tool is used for qualitative queries: risk factor narratives, business descriptions, management commentary, and legal disclosures. It uses the same two-pass filtering strategy as `search_financial_tables` to guarantee company and year isolation.

### 6.3 `python_calculator`

Executes arbitrary Python arithmetic expressions inside a sandboxed AST evaluator. The agent is instructed to use this tool for every calculation, no matter how trivial. This eliminates the well-documented tendency of LLMs to produce plausible but incorrect numerical results when performing arithmetic in free text.

Example invocations:

- Year-over-year growth: `((390 - 383) / 383) * 100` → `1.826...`
- Operating margin: `(85000 / 390000) * 100` → `21.79...`
- Absolute difference: `390000 - 383000` → `7000`

> **Zero Hallucination Arithmetic** — By routing all calculations through a Python AST sandbox rather than letting the LLM compute inline, the system guarantees that every number in the final response is mathematically exact. The LLM's role is reasoning and synthesis — not arithmetic.

---

## 7. End-to-End Data Flow

The following sequence describes the complete path from user input to rendered response. Each step maps to a specific component of the system.

1. User submits a natural-language query via the Streamlit chat interface.
2. `summarize_history` compresses older conversation turns if the session history exceeds `HISTORY_THRESHOLD`.
3. `rewrite_query` decomposes the query into one or more precise sub-questions.
4. `route_after_rewrite` determines whether a single agent pass or multiple sequential passes are required.
5. The agent executes the ReAct loop for each sub-question, calling `search_financial_tables`, `search_unstructured_text`, and `python_calculator` as needed.
6. `aggregate_answers` collects the results from all sub-question passes.
7. The LLM synthesises a final answer with inline citations drawn from the retrieved sources.
8. The Streamlit UI renders the answer in the chat panel and exposes the full Thought → Action → Observation trace in the collapsible expander.

---

## 8. Configuration Reference

All tuneable parameters are centralised in `src/agent/config.py`. Adjust these values to balance response quality against latency and API cost.

| Constant | Default | Effect |
|---|---|---|
| `MAX_ITERATIONS` | `5` | Maximum Thought → Action → Observation cycles per sub-question before the agent is forced to return its best available answer. |
| `MAX_TOOL_CALLS` | `10` | Hard cap on the total number of tool invocations across all iterations. Prevents runaway loops on ambiguous queries. |
| `TOKEN_THRESHOLD` | `4000` | Character count at which intermediate observations are compressed before the next iteration. Prevents context overflow. |
| `GROWTH_FACTOR` | `1.3` | Multiplier applied to the character-based token estimate to account for special tokens and formatting overhead. |
| `HISTORY_THRESHOLD` | `4` | Number of conversation turns after which the session history is summarised rather than passed verbatim. |
| `PRE_FILTER_K` | `20` | Size of the initial candidate set retrieved by semantic similarity before Python-side metadata filtering is applied. |

---

## 9. Known Limitations and Design Decisions

### 9.1 Legacy AgentExecutor Code

The codebase contains a legacy LangChain `AgentExecutor` implementation (`src/agent/generator.py`) that is no longer invoked by the production application. It has been retained as a reference implementation and to preserve git history, but all production traffic is handled by the Tool-Calling Agent described in Section 5. It may be safely removed in a future cleanup pass.

### 9.2 Token Estimation Heuristic

Context size is estimated using the formula `characters / 4`, which approximates the average token length in English text. This heuristic is reliable for narrative sections but may underestimate token counts for large Markdown tables, which contain a high proportion of numbers, pipes, and hyphens. In practice, the `GROWTH_FACTOR` multiplier provides sufficient headroom, but extremely large tables may still cause occasional context overflow in edge cases.

### 9.3 ChromaDB Metadata Filter Workaround

ChromaDB's combined semantic-and-metadata filtering does not always behave deterministically when multiple `where` clauses are combined with a similarity search. The two-pass retrieval strategy — fetch `PRE_FILTER_K` results by similarity, then filter in Python — was introduced as a reliable workaround. The tradeoff is increased memory usage proportional to `PRE_FILTER_K`, which is negligible at the default setting of 20 but should be monitored if the corpus grows significantly.

### 9.4 Sequential Sub-Query Execution

Sub-questions derived from a single user query are currently executed sequentially within the agent loop. A parallel execution strategy would reduce end-to-end latency for multi-part queries but would introduce the risk of hitting OpenAI API rate limits when several tool calls are issued simultaneously. Sequential execution was chosen as the safer default; parallel execution can be introduced once rate-limit handling and retry logic are in place.

---

*Financial Report ChatBot — Application Handbook | Version 1.0*