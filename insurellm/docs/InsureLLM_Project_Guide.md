# InsureLLM --- Project Guide & RAG Walkthrough

> **Audience:** developers, reviewers, engineering managers, and anyone
> onboarding to InsureLLM.\
> **Goal:** explain the purpose of the project, every major component,
> the libraries involved, and exactly how RAG works in this repository.

![InsureLLM architecture](insurllm%20architecture.png)

## 1. Purpose

InsureLLM is an insurance-domain knowledge assistant built to
demonstrate **Retrieval-Augmented Generation (RAG)**. Its private/local
knowledge base contains information about a fictional InsureLLM company:
employees, products, contracts, and company information.

A normal LLM answers from information learned during model training plus
the prompt it receives. InsureLLM instead retrieves relevant passages
from its own knowledge base and inserts those passages into the prompt
before asking the LLM to answer. This makes the answer much more useful
for company-specific questions and reduces dependence on the model's
general pretrained knowledge.

The project demonstrates the complete RAG lifecycle:

1.  load domain documents;
2.  add metadata;
3.  split documents into overlapping chunks;
4.  convert chunks to embeddings;
5.  persist embeddings in Chroma;
6.  embed the user's question;
7.  retrieve semantically similar chunks;
8.  augment the LLM prompt with those chunks;
9.  generate an answer through OpenRouter;
10. expose the workflow through Gradio;
11. evaluate retrieval and answer quality.

## 2. Repository Structure

``` text
insurellm/
├── insurellm_part1.ipynb        # baseline keyword-context prototype
├── insurellm_part2.ipynb        # ingestion, chunking, embeddings, Chroma, t-SNE
├── insurellm_part3.ipynb        # RAG retrieval + LLM + Gradio
├── insurellm_part4.ipynb        # evaluation walkthrough
├── insurellm_part5.ipynb        # currently empty (0 bytes)
├── scraper.py                   # optional website text/link utility
├── implementation/
│   └── answer.py                # reusable RAG implementation
├── evaluation/
│   ├── test.py                  # TestQuestion schema + JSONL loader
│   ├── eval.py                  # retrieval and answer evaluation
│   └── tests.jsonl              # 150 evaluation questions
├── knowledge-base/
│   ├── employees/               # 32 Markdown files
│   ├── products/                # 8 Markdown files
│   ├── contracts/               # 32 Markdown files
│   └── company/                 # 4 Markdown files
├── vector_db/                   # persisted Chroma database/index
├── assets/                      # notebook/project images
├── docs/                        # developer documentation
└── .env                         # local secrets/configuration
```

The knowledge base therefore contains **76 Markdown source documents**.

## 3. Project Components

### 3.1 Part 1 --- Baseline contextual chat

`insurellm_part1.ipynb` demonstrates a simple approach before vector
search. Employee and product documents are loaded into a Python
dictionary. Words from the user's question are matched against
dictionary keys; matching documents are appended to the system prompt.

This is useful as a baseline, but it is not semantic retrieval. A
synonym, paraphrase, spelling variation, or indirect question may not
match the dictionary key.

The notebook uses:

-   `glob` and `pathlib` for file discovery;
-   `python-dotenv` for the OpenRouter key;
-   the `openai` client pointed at OpenRouter;
-   Gradio for a browser chat interface.

### 3.2 Part 2 --- Knowledge ingestion and vector database

`insurellm_part2.ipynb` builds the RAG index.

**Load:** each folder is loaded with `DirectoryLoader` + `TextLoader`.
The folder name becomes `metadata["doc_type"]`, so documents retain the
categories `employees`, `products`, `contracts`, or `company`.

**Split:** documents are divided with:

``` python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

The overlap intentionally repeats some text between neighboring chunks
so information around a chunk boundary keeps useful context.

**Embed:** the active model is:

``` python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

`all-MiniLM-L6-v2` converts each chunk into a **384-dimensional dense
vector**.

**Persist:** chunks, metadata, and vectors are stored through:

``` python
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)
```

The notebook deletes the existing Chroma collection before rebuilding
it.

**Inspect/visualize:** embeddings are read from Chroma, converted to a
NumPy array, reduced to 2-D or 3-D with t-SNE, and displayed with
Plotly. `doc_type` values are mapped to colors so developers can
visually inspect clustering.

### 3.3 Part 3 --- RAG chat

`insurellm_part3.ipynb` reopens the persisted Chroma store with the same
embedding model and converts it to a retriever:

``` python
retriever = vectorstore.as_retriever()
```

It configures `ChatOpenAI` to use OpenRouter's OpenAI-compatible
endpoint and then performs:

``` text
question
→ retriever
→ relevant chunks
→ concatenate chunk text
→ insert text into system prompt
→ send SystemMessage + HumanMessage to LLM
→ return response.content
```

A `gr.ChatInterface` provides the interactive UI.

### 3.4 `implementation/answer.py` --- reusable RAG logic

This is the most important reusable runtime module. The evaluation code
imports it directly.

Configuration in the supplied code:

``` text
Embedding model : all-MiniLM-L6-v2
Vector DB       : vector_db/
Retrieval K     : 10
LLM gateway     : OpenRouter
Default model   : openai/gpt-4o-mini
Temperature     : 0
```

Key functions:

-   `fetch_context(question)` → retrieves relevant `Document` objects.
-   `combined_question(question, history)` → joins prior user turns with
    the current question for retrieval.
-   `answer_question(question, history)` → retrieves context, builds
    messages, invokes the LLM, and returns `(answer, docs)`.

### 3.5 Evaluation

`evaluation/tests.jsonl` contains **150** questions:

  Category           Count
  ---------------- -------
  `direct_fact`         70
  `temporal`            20
  `spanning`            20
  `comparative`         10
  `numerical`           10
  `relationship`        10
  `holistic`            10

`evaluation/test.py` defines the Pydantic `TestQuestion` schema and
loads the JSONL records.

`evaluation/eval.py` measures two different things:

**Retrieval quality** - MRR --- how early a keyword-relevant result
appears; - nDCG --- rank-aware binary relevance; - keyword coverage ---
percentage of expected keywords found in retrieved chunks.

**Answer quality** - calls the real RAG pipeline; - compares the
generated answer with the reference answer; - uses LiteLLM/OpenRouter as
an LLM judge; - returns accuracy, completeness, relevance, and textual
feedback.

### 3.6 `scraper.py`

This optional utility uses `requests` and BeautifulSoup to fetch a
website, remove elements such as scripts/styles/images/inputs, extract
text, truncate it to 2,000 characters, or return links. It is **not part
of the current core RAG ingestion path**.

## 4. Exactly How RAG Works Here

### 4.1 Offline indexing

A Markdown document is loaded into a LangChain `Document`:

``` python
Document(
    page_content="...",
    metadata={
        "source": "...",
        "doc_type": "employees"
    }
)
```

The splitter creates smaller `Document` chunks. The embedding model
converts every chunk into a 384-number vector:

``` text
text chunk
   │
   ▼
all-MiniLM-L6-v2
   │
   ▼
[0.018, -0.041, 0.072, ... 384 values ...]
```

Chroma stores the vector together with the corresponding text and
metadata.

### 4.2 Online retrieval

For a question such as `Who is Avery Lancaster?`, the retriever uses the
**same embedding model** to produce a query vector.

Chroma searches its stored vectors for semantically nearby chunks. In
`implementation/answer.py`, the retriever requests the top **10**
documents.

The embedding model does not generate the answer. Its job is to
represent text numerically so semantic similarity can be calculated.

### 4.3 Augmentation

The retrieved document text is combined:

``` python
context = "\n\n".join(
    doc.page_content for doc in docs
)
```

and inserted into `SYSTEM_PROMPT`.

That injection of retrieved private knowledge into the model prompt is
the **Augmented** part of RAG.

### 4.4 Generation

The final model input consists of:

``` text
SystemMessage
  ├── InsureLLM assistant instructions
  └── retrieved context

converted conversation history

HumanMessage
  └── current question
```

`ChatOpenAI` sends these messages to OpenRouter. The model then
generates the natural-language response.

**Important:** the LLM does not directly search Chroma. Application code
retrieves first, builds the prompt, and only then calls the LLM.

### 4.5 Conversation-aware retrieval

`combined_question()` extracts earlier user messages and joins them with
the current question before retrieval. The full history is also
converted into LangChain messages for generation. This provides a
lightweight way for follow-up questions to carry conversational context.

## 5. Libraries and Why They Exist

  -----------------------------------------------------------------------
  Library                             Purpose in InsureLLM
  ----------------------------------- -----------------------------------
  `langchain-openai`                  `ChatOpenAI` model abstraction;
                                      `OpenAIEmbeddings` is imported as
                                      an alternative in Part 2

  `langchain-chroma`                  LangChain integration for
                                      creating/reopening/querying Chroma

  `langchain-huggingface`             wraps `all-MiniLM-L6-v2` as the
                                      active embedding model

  `langchain-core`                    `Document`, `SystemMessage`,
                                      `HumanMessage`, history conversion

  `langchain-community`               `DirectoryLoader` and `TextLoader`

  `langchain-text-splitters`          recursive text chunking

  `chromadb`                          underlying persistent vector
                                      database

  `openai`                            direct OpenAI-compatible client in
                                      the Part 1 baseline

  `litellm`                           model gateway used by LLM-as-judge
                                      evaluation

  `gradio`                            browser chat UI

  `python-dotenv`                     loads local API/model configuration

  `tiktoken`                          counts knowledge-base tokens for
                                      analysis

  `numpy`                             embedding matrix handling

  `scikit-learn`                      t-SNE dimensionality reduction

  `plotly`                            interactive 2-D/3-D embedding
                                      visualization

  `pydantic`                          typed test/evaluation schemas

  `requests`                          HTTP fetching for the optional
                                      scraper

  `beautifulsoup4`                    HTML parsing/cleanup for the
                                      optional scraper

  `glob`, `pathlib`, `os`, `json`,    file discovery, paths,
  `math`                              configuration, JSONL parsing,
                                      metric calculations
  -----------------------------------------------------------------------

## 6. Configuration and Secrets

The runtime loads `.env`. `implementation/answer.py` recognizes
`OPENROUTER_MODEL` and obtains an API key from `OPENROUTER_API_KEY`,
with `OPENAI_API_KEY` as a fallback.

Do not commit or share real keys. A repository should normally ignore
`.env`, notebook checkpoints, and Python cache files.

## 7. Developer Notes

-   Run/build Part 2 before expecting Part 3 or
    `implementation/answer.py` to retrieve meaningful data.
-   The embedding model used for querying must remain compatible with
    the model used to create the stored vectors.
-   `doc_type` metadata is stored but the current retriever does not use
    a metadata filter.
-   `RETRIEVAL_K` is fixed at 10 in the reusable implementation.
-   There is no reranker, hybrid keyword/vector search, or explicit
    similarity threshold.
-   Retrieved `Document` objects are returned by `answer_question()` but
    the current Gradio response exposes only answer text; source
    citations are a useful next enhancement.
-   Part 2's 3-D hover expression appears to treat Chroma-returned
    document strings as LangChain `Document` objects; correct that cell
    before relying on its hover text.
-   `insurellm_part5.ipynb` is currently empty.
-   `history=[]` is used as a default argument. It is not currently
    mutated, but `None` is safer if the function evolves.

## 8. One-Line Mental Model

``` text
76 Markdown files
→ LangChain Documents
→ 1000/200 chunks
→ MiniLM 384-D embeddings
→ Chroma
→ semantic top-10 retrieval
→ retrieved context in system prompt
→ OpenRouter LLM
→ answer
→ optional retrieval + LLM-judge evaluation
```

LangChain is the **integration/orchestration layer**; Chroma is the
**vector store**; MiniLM is the **embedding model**; and the
OpenRouter-selected chat model is the **answer generator**.
