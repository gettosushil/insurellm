# InsureLLM --- Technical Design & Developer Specification

**Purpose:** implementation-level architecture and onboarding reference
for the InsureLLM RAG project.\
**Scope:** documents the code and data present in the supplied
repository.

![InsureLLM architecture](insurllm%20architecture.png)

## 1. Architecture Summary

InsureLLM separates the system into an **offline indexing path**, an
**online RAG path**, and an **evaluation path**.

``` text
OFFLINE
knowledge-base/*.md
 → DirectoryLoader/TextLoader
 → doc_type metadata
 → RecursiveCharacterTextSplitter (1000/200)
 → HuggingFace all-MiniLM-L6-v2
 → 384-D embeddings
 → Chroma vector_db/

ONLINE
question + history
 → combined retrieval query
 → same embedding model
 → Chroma retriever (k=10)
 → relevant Documents
 → SYSTEM_PROMPT + context + history + question
 → ChatOpenAI
 → OpenRouter
 → configured LLM
 → answer + retrieved Documents

EVALUATION
tests.jsonl
 → retrieval metrics: MRR / nDCG / keyword coverage
 → answer metrics: accuracy / completeness / relevance
 → LiteLLM/OpenRouter LLM judge
```

## 2. Component Contracts

  ----------------------------------------------------------------------------------
  Component               Location                     Contract / responsibility
  ----------------------- ---------------------------- -----------------------------
  Knowledge source        `knowledge-base/`            76 UTF-8 Markdown files
                                                       grouped by domain

  Baseline                `insurellm_part1.ipynb`      literal-keyword context
                                                       injection + direct
                                                       OpenRouter-compatible client

  Index builder           `insurellm_part2.ipynb`      load, tag, chunk, embed,
                                                       persist, inspect and
                                                       visualize

  RAG demo                `insurellm_part3.ipynb`      reopen vector store,
                                                       retrieve, prompt, generate,
                                                       Gradio UI

  Runtime RAG module      `implementation/answer.py`   reusable retrieval and answer
                                                       functions

  Test model              `evaluation/test.py`         typed `TestQuestion` + JSONL
                                                       loading

  Evaluation engine       `evaluation/eval.py`         deterministic retrieval
                                                       metrics + LLM-as-judge

  Test fixtures           `evaluation/tests.jsonl`     150 questions/reference
                                                       answers/keywords/categories

  Vector persistence      `vector_db/`                 Chroma-managed SQLite/index
                                                       files

  Scraper utility         `scraper.py`                 optional HTTP text/link
                                                       extraction

  Configuration           `.env`                       local secrets/model
                                                       configuration
  ----------------------------------------------------------------------------------

## 3. Data Model

### 3.1 Source classification

The knowledge base currently contains:

  `doc_type`      Files
  ------------- -------
  `employees`        32
  `products`          8
  `contracts`        32
  `company`           4

During ingestion:

``` python
doc.metadata["doc_type"] = doc_type
```

The source path supplied by the loader and this category metadata travel
with each chunk.

### 3.2 Chunking contract

``` python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

The splitter recursively attempts sensible separators rather than
blindly cutting every 1,000 characters. The overlap reduces
boundary-related information loss.

Changing chunk size/overlap changes the retrieval corpus and therefore
requires rebuilding the vector store.

### 3.3 Embedding contract

``` python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

The model produces **384-dimensional dense embeddings**. Index-time and
query-time embeddings must use the same vector space.

`OpenAIEmbeddings(model="text-embedding-3-large")` appears only as a
commented alternative in Part 2; it is not the active persisted-vector
configuration.

### 3.4 Chroma contract

Index creation:

``` python
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)
```

Runtime reopening:

``` python
Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)
```

Application code should use the Chroma API rather than directly
manipulating `chroma.sqlite3` or UUID index files.

## 4. Online RAG Specification

### 4.1 Retriever

`implementation/answer.py` configures:

``` python
RETRIEVAL_K = 10

retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)
```

`fetch_context(question)` returns `list[Document]`.

No metadata filter, reranking stage, keyword fusion, or explicit
similarity threshold is currently configured.

### 4.2 Conversational query

``` python
combined_question(question, history)
```

joins earlier **user-role** messages with the current question for
retrieval. This gives the retriever more context for follow-up questions
without introducing a separate query-rewriting LLM.

### 4.3 Context construction

``` python
context = "\n\n".join(
    doc.page_content for doc in docs
)
```

The result is interpolated into `SYSTEM_PROMPT`.

### 4.4 Message construction

The generation call receives:

1.  a `SystemMessage` containing assistant instructions plus retrieved
    context;
2.  converted chat history from `convert_to_messages(history)`;
3.  a `HumanMessage` containing the current question.

This separation is important: **retrieval and generation are distinct
operations**.

### 4.5 LLM gateway

The reusable module uses:

``` python
ChatOpenAI(
    model=MODEL,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "llm_engineering"
    }
)
```

`ChatOpenAI` supplies the LangChain chat-model interface, while the
overridden `base_url` sends the OpenAI-compatible request through
OpenRouter.

Model selection:

``` python
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)
```

### 4.6 Output contract

``` python
answer_question(...) -> tuple[str, list[Document]]
```

The tuple contains: - generated answer text; - the exact retrieved
documents used to build context.

This is valuable for debugging/evaluation and can later support source
citations.

## 5. Library-Level Design

  -----------------------------------------------------------------------------------------------
  Library/module                                  Design role             Exact project use
  ----------------------------------------------- ----------------------- -----------------------
  `langchain_openai.ChatOpenAI`                   chat-model adapter      invokes an
                                                                          OpenAI-compatible model
                                                                          through OpenRouter

  `langchain_openai.OpenAIEmbeddings`             alternative embedder    imported/commented in
                                                                          Part 2; not active

  `langchain_chroma.Chroma`                       vector-store adapter    index creation,
                                                                          persistence, reopening,
                                                                          retriever

  `langchain_huggingface.HuggingFaceEmbeddings`   embedding adapter       MiniLM document/query
                                                                          embeddings

  `langchain_core.documents.Document`             domain object           retrieved chunk +
                                                                          metadata contract

  `langchain_core.messages`                       message model           system/user/history
                                                                          message construction

  `langchain_community.document_loaders`          ingestion               directory discovery and
                                                                          UTF-8 text loading

  `langchain_text_splitters`                      preprocessing           recursive 1000/200
                                                                          chunking

  `chromadb`                                      storage/search engine   local vector
                                                                          persistence underneath
                                                                          LangChain

  `openai`                                        direct compatible API   Part 1 baseline
                                                  client                  

  `litellm`                                       evaluation model        OpenRouter LLM-as-judge
                                                  abstraction             call

  `pydantic`                                      typed validation        `TestQuestion`,
                                                                          `RetrievalEval`,
                                                                          `AnswerEval`

  `gradio`                                        UI                      `ChatInterface`
                                                                          prototype

  `python-dotenv`                                 configuration           `.env` loading

  `tiktoken`                                      analysis                knowledge-base token
                                                                          counting

  `numpy`                                         vector processing       embedding matrix
                                                                          creation

  `scikit-learn`                                  dimensionality          t-SNE
                                                  reduction               

  `plotly`                                        visualization           interactive 2-D/3-D
                                                                          embedding plots

  `requests` + `beautifulsoup4`                   optional ingestion      website
                                                  utility                 fetch/cleanup/link
                                                                          extraction
  -----------------------------------------------------------------------------------------------

## 6. Evaluation Design

### 6.1 Test fixture

Each JSONL record maps to:

``` python
class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str
```

The supplied file has 150 tests: 70 direct-fact, 20 temporal, 20
spanning, and 10 each comparative, numerical, relationship, and
holistic.

### 6.2 Retrieval metrics

`calculate_mrr(keyword, retrieved_docs)` scans retrieved documents in
rank order and returns `1/rank` for the first case-insensitive keyword
match.

`calculate_ndcg(keyword, retrieved_docs, k=10)` builds binary relevance
values, computes DCG, computes ideal DCG, and returns normalized DCG.

`evaluate_retrieval()` averages keyword-level MRR/nDCG and calculates
keyword coverage.

These metrics test whether retrieval found expected evidence; they do
not by themselves prove semantic correctness.

### 6.3 Answer metrics

`evaluate_answer()`:

1.  generates a real RAG answer;
2.  constructs a judge prompt with question, generated answer, and
    reference answer;
3.  calls LiteLLM with `openrouter/openai/gpt-4o-mini`;
4.  requests structured `AnswerEval`;
5.  returns feedback plus 1--5 scores for accuracy, completeness, and
    relevance.

This layer is slower and model-dependent compared with deterministic
retrieval metrics.

## 7. Visualization Design

Part 2 reads Chroma records with:

``` python
collection.get(
    include=["embeddings", "documents", "metadatas"]
)
```

Embeddings become a NumPy matrix; metadata provides `doc_type`; t-SNE
reduces 384 dimensions to 2 or 3; Plotly displays the points.

The visualization is exploratory. t-SNE axes have no direct semantic
meaning and should not be used as retrieval scores.

## 8. Configuration & Security

The code loads `.env` using `python-dotenv`. Relevant variables include:

``` text
OPENROUTER_API_KEY
OPENROUTER_MODEL
OPENAI_API_KEY       # fallback in implementation/answer.py
```

Recommended `.gitignore`:

``` text
.env
.env.*
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Do not expose real API keys in notebooks, documentation, logs,
screenshots, or commits.

## 9. Known Design Constraints

1.  top-k is fixed at 10;
2.  no similarity threshold;
3.  no metadata filtering despite `doc_type` being available;
4.  no lexical/vector hybrid retrieval;
5.  no cross-encoder or LLM reranking;
6.  no explicit source citations in the current chat response;
7.  prompt grounding is advisory rather than strict;
8.  re-indexing is notebook-driven rather than an automated ingestion
    service;
9.  Part 2 deletes the current Chroma collection before rebuilding;
10. changing embedding models requires rebuilding the vector database;
11. `history=[]` is a mutable-default style risk if future code mutates
    it;
12. Part 2's 3-D hover expression appears inconsistent with the string
    documents returned by `collection.get`;
13. `insurellm_part5.ipynb` is an empty placeholder.

## 10. Recommended Production Evolution

``` text
UI / API
   │
RAG service
   ├── standalone-query rewriting
   ├── metadata-aware retrieval
   ├── hybrid semantic + lexical search
   ├── reranking
   ├── relevance threshold / abstention
   ├── source citations
   └── tracing, latency and token metrics
          │
versioned vector/document store
          │
automated ingestion + validation pipeline
```

Recommended engineering work: - move ingestion from notebooks into a
tested Python module/CLI; - add dependency locking
(`pyproject.toml`/lockfile or pinned requirements); - add `.env.example`
with no secrets; - add unit tests around retrieval, prompt construction,
and configuration; - add document/chunk IDs and display source
citations; - benchmark multiple `k`, chunk sizes, and embedding
models; - add prompt-injection defenses if source documents can be
untrusted; - version the index by embedding model and chunking
configuration.

## 11. New-Developer Reading Order

1.  `docs/InsureLLM_Project_Guide.md`
2.  `docs/insurllm architecture.png`
3.  `insurellm_part2.ipynb`
4.  `implementation/answer.py`
5.  `insurellm_part3.ipynb`
6.  `evaluation/test.py` and `tests.jsonl`
7.  `evaluation/eval.py`
8.  `insurellm_part4.ipynb`

## 12. Canonical Flow

``` text
76 source Markdown files
→ loader + doc_type metadata
→ recursive 1000-character chunks / 200 overlap
→ all-MiniLM-L6-v2
→ 384-dimensional embeddings
→ local Chroma persistence
→ top-10 semantic retrieval
→ retrieved page_content
→ system prompt + history + current question
→ ChatOpenAI via OpenRouter
→ generated answer
→ retrieval metrics and optional LLM-as-judge
```
