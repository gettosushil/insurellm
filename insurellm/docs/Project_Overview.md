# InsureLLM — Project Overview

> **Expert Knowledge Worker for InsureLLM Insurance Tech**
>
> A low-cost, high-accuracy Retrieval-Augmented Generation (RAG) system that lets employees and stakeholders ask natural-language questions about the company, its people, products, and contracts — and get grounded, cited answers.

---

## 1. Purpose & Vision

### 1.1 The Business

**InsureLLM** is a fictional Insurance Tech company (San Francisco HQ) used as a realistic enterprise testbed:

| Fact | Value |
|------|-------|
| Founded | 2015 by **Avery Lancaster** |
| Journey | 200 employees (2020 peak) → strategic restructuring (2022-23) → **32 employees** (2025) |
| Footprint | HQ San Francisco + satellites NYC / Austin / Chicago / Denver, remote-first |
| Portfolio | **8 products**: `Markellm` (marketplace, first product), `Carllm`, `Homellm`, `Rellm`, `Lifellm`, `Healthllm`, `Bizllm`, `Claimllm` |
| Scale | **32 active contracts** across all product lines, from regional carriers to global reinsurers |

### 1.2 The Problem

* Knowledge is fragmented across 80+ markdown documents (company strategy, employee bios, product specs, legal contracts).
* A vanilla LLM hallucinates: it invents employee awards, contract dates, or product features.
* Employees need an **expert knowledge worker** — fast, accurate, cheap, and auditable.

### 1.3 The Solution

A **RAG pipeline** that:

1. Chunks and embeds every knowledge-base document into a vector store **once** (offline).
2. At query time retrieves the *most relevant* chunks (semantic search, not keyword match).
3. Injects them as context into a grounded system prompt and generates an answer with an OpenRouter LLM.
4. Evaluates retrieval *and* answer quality on a curated test suite, so progress is measurable.

**Design goals:** `accuracy > creativity`, `cost < $0.01 / query`, `reproducibility`, `local-first vectors`.

---

## 2. Project Structure

```
insurellm/
├── knowledge-base/          # Source of truth — 80+ Markdown documents
│   ├── company/             # about.md, careers.md, culture.md, overview.md (4 files)
│   ├── employees/           # 32 bios, e.g. Avery Lancaster.md, Alex Chen.md
│   ├── products/            # 8 product sheets: Bizllm.md ... Rellm.md
│   └── contracts/           # 32 anonymised contract summaries
├── vector_db/               # Chroma persistent store (chroma.sqlite3 ≈ 5 MB)
│   └── chroma.sqlite3       # 384-dim vectors, created by part2.ipynb
├── implementation/
│   └── answer.py            # Production RAG module: fetch_context() + answer_question()
├── evaluation/
│   ├── tests.jsonl          # ~50 curated Q/A pairs (question, keywords, reference_answer, category)
│   ├── test.py              # Pydantic TestQuestion model + load_tests()
│   └── eval.py              # Retrieval (MRR, nDCG) + LLM-as-judge (accuracy/completeness/relevance)
├── assets/                  # Illustrations for notebooks (business.jpg, core.jpg …)
├── scraper.py               # Helpers fetch_website_contents / fetch_website_links (BeautifulSoup)
├── insurellm_part1.ipynb    # Day 1 — Keyword baseline + Gradio chat
├── insurellm_part2.ipynb    # Day 2 — Chunking → Embeddings → Chroma → t-SNE visualisation
├── insurellm_part3.ipynb    # Day 3 — LangChain RAG + OpenRouter ChatOpenAI
├── insurellm_part4.ipynb    # Day 4 — Evaluation harness (retrieval + answer)
├── docs/
│   ├── Project_Overview.md          # ← This file
│   └── Technical_Specification.md   # Architecture & onboarding for developers
└── .env                     # OPENROUTER_API_KEY, OPENROUTER_MODEL
```

---

## 3. Components in Detail

### 3.1 Knowledge Base (`knowledge-base/`)

The single source of truth. All files are Markdown, UTF-8, with front-matter-free prose — intentionally diverse in length and style to mimic a real enterprise wiki.

* **company/** — narrative history, headcount evolution, office strategy, values.
* **employees/** — 32 profiles (role, tenure, awards, e.g. *Maxine Thompson won IIOTY 2023*).
* **products/** — 8 product pages with launch dates, contract counts, positioning.
* **contracts/** — legal summaries used to test the system on sensitive, high-value retrieval.

> The RAG system never trains on this data — it *retrieves* it at query time, so updates are immediate after re-ingestion.

### 3.2 Baseline Chat — `insurellm_part1.ipynb`

Purpose: show why naive approaches fail and establish a floor to beat.

* Loads all `employees/*.md` + `products/*.md` into an in-memory `dict knowledge` (`name.lower() -> content`).
* `get_relevant_context(message)` — naive keyword filter: `word in knowledge`.
* `SYSTEM_PREFIX + additional_context(message)` + `OpenAI(base_url=openrouter.ai/api/v1)` → answer.
* `gr.ChatInterface(chat).launch()` for quick prototyping.

**Lesson:** keyword matching misses paraphrases ("founder" vs "started the company") and pulls in irrelevant chunks.

### 3.3 Ingestion & Vector Store — `insurellm_part2.ipynb`

This is the **offline** build step. It runs once (or on knowledge-base changes).

| Step | Code | Detail |
|------|------|--------|
| **Count** | `glob("knowledge-base/**/*.md")` + `tiktoken.encoding_for_model` | ~80 files, reports total chars/tokens for cost planning |
| **Load** | `DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader)` | One `Document` per file, adds `metadata["doc_type"]` = company/employees/products/contracts |
| **Split** | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` | ~300 chunks; overlap preserves cross-boundary context. Tries `\n\n` → `\n` → ` ` → `` |
| **Embed** | `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` | 384-dim, local, free, no API cost. Alternative `OpenAIEmbeddings(text-embedding-3-large)` commented out for comparison |
| **Store** | `Chroma.from_documents(chunks, embedding, persist_directory="vector_db")` | Persistent Chroma; `delete_collection()` if re-running |
| **Verify** | `collection.count()`, `collection.get(include=["embeddings"])` | Logs `~300 vectors × 384 dims` |
| **Visualise** | `TSNE(n_components=2/3) + plotly.graph_objects` | 2D & 3D scatter, coloured by `doc_type` (blue=products, green=employees, red=contracts, orange=company). Hover shows first 100 chars. Proves semantic clustering. |

> **Why 1000/200?** Balances LLM context window cost (gpt-4.1-nano ≈ 128k) with answer completeness. 1000 chars ≈ 250 tokens; 10 chunks ≈ 2500 tokens of context — cheap and focused.

### 3.4 RAG Pipeline — `insurellm_part3.ipynb` + `implementation/answer.py`

The **online** path (per user query):

```
User question
    → retriever.invoke(question, k=10)         [Chroma → cosine similarity]
    → context = "\n\n".join(doc.page_content)   [10 chunks ≈ 10k chars]
    → SYSTEM_PROMPT.format(context=context)     [grounded instruction]
    → ChatOpenAI(model, api_key, base_url)      [OpenRouter, temperature=0]
    → answer + retrieved docs
```

* `retriever = vectorstore.as_retriever(search_kwargs={"k": 10})`
* `SYSTEM_PROMPT` explicitly says *“If relevant, use the given context … If you don’t know, say so.”* — reduces hallucination.
* `combined_question(history)` concatenates prior user messages for multi-turn context.
* `langchain_core.messages.convert_to_messages(history)` preserves role fidelity.
* `temperature=0` = most deterministic; real creativity comes from prompt, not sampling.

### 3.5 Evaluation — `insurellm_part4.ipynb` + `evaluation/`

Evaluation is a first-class citizen, not an afterthought.

**Dataset (`tests.jsonl`):** ~50 questions across categories `direct_fact`, `spanning`, `temporal`, `comparison`. Each record:

```json
{"question": "Who won the prestigious IIOTY award in 2023?",
 "keywords": ["Maxine","Thompson","IIOTY"],
 "reference_answer": "Maxine Thompson won ... 2023.",
 "category": "direct_fact"}
```

**`test.py`:** `TestQuestion(BaseModel)` (pydantic) + `load_tests()`.

**`eval.py`:**

| Metric | What it measures | How computed |
|--------|------------------|--------------|
| **MRR** (Mean Reciprocal Rank) | How high the first relevant doc appears | `1/rank` per keyword, averaged |
| **nDCG@k** (Normalised DCG) | Rank-aware relevance (binary) | DCG / ideal DCG, k=10 |
| **Keyword coverage** | % of keywords found in top-k | `found / total * 100` |
| **LLM-as-judge** | Accuracy / Completeness / Relevance (1-5) | `litellm.completion(model="gpt-4.1-nano")` with `AnswerEval` structured output, comparing `generated_answer` vs `reference_answer` |

```python
from evaluation.eval import evaluate_retrieval, evaluate_answer
t = test.load_tests()[0]
evaluate_retrieval(t)          # → RetrievalEval(mrr=0.16, ndcg=0.29, ...)
answer, docs = answer_question(t.question)
evaluate_answer(t)             # → AnswerEval(accuracy=5, completeness=4, relevance=5)
```

### 3.6 Utilities

* **`scraper.py`** — `fetch_website_contents(url)` strips `script/style/img/input` and truncates to 2000 chars; `fetch_website_links(url)` enumerates `<a href>`. Used for optional web augmentation, not core RAG.
* **`.env`** — `OPENROUTER_API_KEY=sk-or-v1-...`, `OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b:...` (or `openai/gpt-4o-mini`). Loaded via `python-dotenv` with `override=True` and explicit `Path(__file__).parent.parent/.env`.

---

## 4. Libraries — At a Glance

| Library | Role in this Project | How Used |
|---------|----------------------|----------|
| **openai** | OpenRouter-compatible LLM client | `OpenAI(base_url="https://openrouter.ai/api/v1", api_key=...)` in part1; underlying client for `langchain-openai` |
| **langchain-openai** | `ChatOpenAI` & `OpenAIEmbeddings` wrappers | `ChatOpenAI(model, api_key, base_url)` for grounded generation (part3, implementation) |
| **langchain-chroma** | Chroma vector store integration | `Chroma(persist_directory, embedding_function)` + `as_retriever()` |
| **langchain-huggingface** | Local embeddings | `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` — 384-dim, free, offline |
| **langchain-community** | Document loaders | `DirectoryLoader`, `TextLoader` for `knowledge-base/**/*.md` |
| **langchain-text-splitters** | Chunking | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` |
| **langchain-core** | Message & document primitives | `SystemMessage`, `HumanMessage`, `convert_to_messages`, `Document` |
| **chromadb** | Persistent vector database | Backing store for `langchain-chroma`; `chroma.sqlite3` |
| **tiktoken** | Token counting | `tiktoken.encoding_for_model(MODEL).encode(text)` to estimate cost before embedding |
| **python-dotenv** | Secrets management | `load_dotenv(dotenv_path=..., override=True)` for `OPENROUTER_API_KEY` |
| **gradio** | Chat UI | `gr.ChatInterface(chat).launch(inbrowser=True)` rapid prototype in part1 |
| **scikit-learn** | Dimensionality reduction | `TSNE(n_components=2/3)` for vector visualisation |
| **plotly** | Interactive plots | `go.Scatter`, `go.Scatter3d` — hover text shows chunk type + preview |
| **litellm** | LLM-as-judge | `completion(model="gpt-4.1-nano", response_format=AnswerEval)` structured evaluation |
| **pydantic** | Schema & validation | `TestQuestion`, `RetrievalEval`, `AnswerEval`, `RankOrder` models |
| **numpy** | Vector math | `np.array(result['embeddings'])` for t-SNE input |
| **beautifulsoup4 + requests** | Web scraping | `scraper.py` — `BeautifulSoup(response.content, "html.parser")` |
| **gradio’s deps (uvicorn, rich, protobuf)** | Runtime | Indirect; version-pinned conflicts noted, non-blocking |

> See **Technical Specification** for deep-dive on *why* each library was chosen vs alternatives, version notes, and exact import paths.

---

## 5. How the RAG Works — End-to-End (Concrete Example)

**Question:** *“Who won the prestigious IIOTY award in 2023?”*

1. **Ingest (once):** `culture.md` is split; the chunk *“Maxine Thompson won the prestigious Insurellm Innovator of the Year (IIOTY) award in 2023 …”* is embedded as a 384-dim vector and stored in Chroma with `metadata={"source": "culture.md", "doc_type": "company"}`.

2. **Embed query:** The same `all-MiniLM-L6-v2` model embeds the question into a 384-dim vector.

3. **Retrieve:** Chroma cosine-search returns top 10 chunks. The culture.md chunk ranks #2. `fetch_context()` returns `list[Document]` (page_content + metadata).

4. **Build prompt:**

   ```
   SYSTEM: You are a knowledgeable, friendly assistant representing Insurellm ...
           Context:
           Extract from culture.md:
           Maxine Thompson won the prestigious Insurellm Innovator of the Year (IIOTY) award ...

           Extract from employees/Maxine Thompson.md:
           Maxine Thompson — Senior Claims Lead, joined 2018 ...

           (8 more chunks …)

   USER: Who won the prestigious IIOTY award in 2023?
   ```

5. **Generate:** `ChatOpenAI(temperature=0, model="nvidia/nemotron-3-ultra-550b:free", base_url=openrouter)` produces *“Maxine Thompson won the IIOTY award in 2023.”* No hallucinated name, because context contained the answer.

6. **Evaluate:** `evaluate_retrieval` checks keywords `["Maxine","Thompson","IIOTY"]` appear in top-10 (MRR 0.5 if at rank 2). `evaluate_answer` asks an LLM judge to score accuracy/completeness/relevance 1-5 against the reference answer.

**If the answer isn’t in the knowledge base**, the system prompt’s *“If you don’t know, say so”* triggers a safe abstention rather than a fabrication — the key reliability win of RAG.

---

## 6. Running the Project

```bash
# 1. Environment
/opt/anaconda3/bin/python -m pip install -q \
  langchain-openai langchain-chroma langchain-huggingface \
  langchain-community langchain-text-splitters chromadb \
  tiktoken gradio python-dotenv scikit-learn plotly litellm pydantic

# 2. Secrets
cp .env.example .env   # then set OPENROUTER_API_KEY=sk-or-v1-... 

# 3. Ingest (creates vector_db/)
jupyter lab insurellm_part2.ipynb  # Run all → vectorstore created

# 4. Chat
jupyter lab insurellm_part1.ipynb  # or part3.ipynb → Gradio on http://127.0.0.1:7860

# 5. Evaluate
jupyter lab insurellm_part4.ipynb
# or
python -c "from evaluation.eval import evaluate_retrieval; from evaluation import test; print(evaluate_retrieval(test.load_tests()[0]))"
```

---

## 7. Limitations & Next Steps

* **Embedding choice:** `all-MiniLM-L6-v2` is cheap but underperforms `text-embedding-3-large` on nuanced legal contracts. Benchmark both; consider hybrid (local for ingestion, OpenAI for high-value queries).
* **Chunking:** Fixed 1000/200 misses table structure; try `MarkdownHeaderTextSplitter` or semantic chunking.
* **Re-ranking:** Current `k=10` raw cosine search; add a cross-encoder re-ranker (`pro_implementation` prototype exists) to improve MRR.
* **Multi-turn:** `combined_question()` is simplistic; switch to query-rewriting via LLM (`pro_implementation.rewrite_query`).
* **Observability:** Add `langsmith` tracing; log retrieval latency, token usage, judge scores per category.

---

*Maintained for onboarding and stakeholder review. For implementation details, API contracts, and architecture diagram, see `Technical_Specification.md`.*
