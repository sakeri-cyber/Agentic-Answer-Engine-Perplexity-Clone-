# Engineering Notes & Architecture Decisions

## 1. Localhost Routing & Docker Networks
* **Issue:** Initial connections to Dockerized Elasticsearch on macOS resulted in `TimeoutError`. Python's `aiohttp` defaulted to routing `localhost` through IPv6 (`::1`), while Docker mapped the exposed ports to the IPv4 loopback (`127.0.0.1`).
* **Resolution:** Hard-pinned all asynchronous database clients (Qdrant and Elasticsearch) to `127.0.0.1` to enforce IPv4 routing and bypass the IPv6 black hole.

## 2. Asynchronous Event Loop Management
* **Issue:** Heavy ML embedding tasks (`embedder.encode()`) blocking the event loop caused idle database connections to time out before the payload was ready.
* **Resolution:** Restructured the ingestion worker (`worker.py`) to execute all vector math *before* initializing the database connections, utilizing Just-In-Time client initialization to preserve connection health.

## 3. Scaling & Real-Time Ingestion Strategy
Currently, the pipeline ingests the top 10 most recent ArXiv papers in a single run. To scale this into a real-time production system, the following architecture is planned:
* **Batch Embedding:** Transitioning from an iterative loop to batch matrix processing `embedder.encode(chunks)` to parallelize model inference.
* **Cron/Task Scheduling:** Wrapping `ingestion.py` in a Celery worker or GitHub Action scheduled via CRON to execute hourly.
* **Idempotent Upserts:** Deduplication is strictly enforced. `uuid5` hashes are generated using the unique ArXiv ID + Chunk Index. These hashes are used as the primary `_id` in both Qdrant and Elasticsearch, ensuring that chronological polling safely overwrites existing documents without duplicating context.

## 4. Hybrid Retrieval & Elasticsearch Watermark Triage
* **Issue:** When executing searches after recreating the Elasticsearch index, the client threw an `ApiError(503, 'search_phase_execution_exception')`. This was caused by the Docker Desktop virtual disk hitting the 85% low-disk watermark threshold, causing Elasticsearch to leave shards unassigned.
* **Resolution:** Executed a cluster settings update payload via API to disable the disk threshold:
```bash
curl -X PUT "[http://127.0.0.1:9200/_cluster/settings](http://127.0.0.1:9200/_cluster/settings)" -H 'Content-Type: application/json' -d'{"persistent": {"cluster.routing.allocation.disk.threshold_enabled": false}}'
```


## 5. Context Fusion & LLM Prompt Generation
* **Milestone:** Developed `app.py` to act as the central Answer Engine interface. The script successfully executes asynchronous hybrid searches across Qdrant and Elasticsearch, deduplicates results via a `fused_context` dictionary, and structures a hallucination-resistant prompt. 
* **Next Step:** Integrating an LLM inference provider (Groq) to pass the generated prompt into an open-weights model for final answer synthesis.



## 6. Upgrading to Enterprise Microservice (FastAPI, HyDE, & RRF)
* **Milestone:** Transitioned the validated `app.py` terminal logic into a production-grade asynchronous web service using FastAPI (`api.py`).
**Architecture Upgrades:**
* **LLM Traffic Routing (HyDE):** Integrated an incredibly fast open-weights LLM (`openai/gpt-oss-20b` via Groq) to intercept raw user queries. The model outputs a structured JSON payload containing isolated lexical keywords for Elasticsearch and a Hypothetical Document Embedding (HyDE) abstract for Qdrant, drastically reducing semantic mismatch.
* **Reciprocal Rank Fusion (RRF):** Implemented the RRF algorithm to normalize and fuse the disparate scoring mechanisms of Cosine Distance (Qdrant) and BM25 (Elasticsearch).
* **Parallel Execution:** Leveraged `asyncio.gather` to ensure both database clusters are queried concurrently, maintaining sub-200ms retrieval latencies before fusion.


## 7. Escaping API Fragility: Hardening the LLM Traffic Cop
* **The Conflict:** To route user queries effectively, the system utilizes a fast LLM to generate Hypothetical Document Embeddings (HyDE) and extract sparse keywords. However, enforcing structured JSON output from an LLM is notoriously fragile. The initial implementation utilized Groq's strict API-level JSON mode (`response_format={"type": "json_object"}`). Because generative models are inherently "chatty" and often prepend markdown blocks (e.g., ` ```json `), the strict API validator repeatedly crashed, throwing a `400 json_validate_failed` error and triggering the engine's safe-fallback mode.

* **The Resolution:** To harden the microservice for production, I engineered a three-tier bypass system:
1. **API Bypass:** Removed the strict API-level JSON enforcer to prevent 400-level HTTP crashes.
2. **Determinism:** Dropped the model inference temperature to `0.0` and routed the task specifically to `llama-3.1-8b-instant` to enforce maximum adherence.
3. **Regex Extraction:** Implemented a robust Python-level Regular Expression (`re.search(r'\{.*\}', ... re.DOTALL)`) to mathematically extract the JSON dictionary from the raw LLM output, completely isolating the data from any hallucinated conversational text or markdown formatting.

* **Result:** The routing layer achieved 100% parse stability, successfully outputting discrete Qdrant and Elasticsearch query payloads.


## 8. The "Cold Start" Telemetry Problem & DuckDB
**The Conflict:** Standard text-matching engines (Qdrant + Elasticsearch) only understand vocabulary overlap, not "quality." To build a true Search Engine, I needed to train a Learning-to-Rank (LTR) Machine Learning model (XGBoost LambdaMART) to re-rank results based on metadata like Citation Velocity, Content Readability, and Semantic Ratios. However, training an LTR model requires millions of rows of user click-logs (telemetry). As a solo developer building from scratch, I had a severe "Cold Start" problem: zero users, meaning zero training data.

**The Resolution:** I engineered a synthetic telemetry pipeline utilizing an "LLM-as-a-Judge" architecture. 
1. **Organic Query Synthesis:** Instead of hardcoding prompts, I used `llama-3.1-8b-instant` to read the ingested ArXiv papers and organically generate three types of queries per paper: layman keyword searches, natural language questions, and highly technical PhD-level queries.
2. **Feature Engineering:** Passed the documents through a custom Python pipeline to calculate complex ranking features (e.g., Flesch-Kincaid readability via `textstat`, simulated author `h_index`, and Title/Body Sparse Divergence).
3. **The LLM Judge:** Triggered Llama-3 to grade the relevance of document candidates against the synthetic queries on a strict 0 to 4 scale using Chain-of-Thought reasoning.
4. **Out-of-Core Processing:** To ensure this pipeline could scale to millions of rows without triggering Out-Of-Memory (OOM) Pandas crashes, I bypassed in-memory dataframes and wrote the pipeline to stream the synthesized feature matrix directly into **DuckDB**—a high-performance, columnar analytical database.

## 8.1. Infrastructure Friction: Overcoming NLTK SSL Certificate Failures
**The Conflict:** To calculate the Flesch-Kincaid readability score for the Learning-to-Rank feature matrix, the pipeline relies on the `textstat` library, which internally uses the Natural Language Toolkit (`nltk`) to count syllables via the Carnegie Mellon University Pronouncing Dictionary (`cmudict`). However, during the pipeline's first run, macOS's strict SSL certificate policies blocked the automated background download of the `cmudict` corpus, throwing an `[SSL: CERTIFICATE_VERIFY_FAILED]` fatal error and crashing the data generation.

**The Resolution:** Rather than altering system-wide macOS security configurations, I engineered a targeted, temporary bypass. I opened an interactive Python shell and monkey-patched the `ssl` module to force an unverified HTTPS context (`ssl._create_unverified_context`). With the strict verification temporarily disabled in memory, I manually triggered `nltk.download('cmudict')`. Once the corpus was securely downloaded to the local environment, the synthetic data pipeline was able to resume and calculate readability features perfectly.


## 8.2. Infrastructure Friction: Multi-Threading & macOS C++ Dependencies
**The Conflict:** When initializing the XGBoost `Booster` for the LambdaMART training loop, the script suffered a fatal core dump (`Library not loaded: @rpath/libomp.dylib`). XGBoost achieves its blazing speed via Open Multi-Processing (OpenMP) to parallelize tree construction across CPU cores. However, Apple's Clang compiler natively omits OpenMP on macOS environments, causing the Python C-API wrapper to fail upon instantiation.

**The Resolution:** Diagnosed the missing system-level C++ dependency and patched the local macOS environment by injecting the `libomp` runtime via Homebrew (`brew install libomp`). This allowed the XGBoost binaries to correctly link to the multi-threading runtime, enabling out-of-core parallelized training.


## 9. Phase 4: XGBoost LambdaMART Ranker Architecture
**The Goal:** Move beyond simple semantic similarity scores by training a machine learning re-ranking model that considers document metadata (citation counts, publication age, text readability) alongside retrieval scores.
**The Implementation:**
* **Training Data Generation:** Built a synthetic data loop using DuckDB to simulate query-document pairs, modeling a non-linear objective function where high dense/sparse scores coupled with modern citation velocities yield a high ground-truth relevance label ($Relevance = 3$).
* **Model Training:** Utilized XGBoost's specialized `rank:pairwise` LambdaMART objective function. Structured the data using `XGBRegressor` / `DMatrix` groups to train the model to sort candidate lists relative to specific query groups rather than predicting absolute scores.

## 10. Phase 5: Modular Live Inference Engine (`ranker.py`)
**The Goal:** Bridge the trained offline XGBoost model with the live production FastAPI server.
**The Implementation:** Created a decoupled inference file (`ranker.py`) that handles real-time feature extraction for runtime search results. 
* It takes incoming RRF candidate documents and immediately calculates a 9-feature live matrix matching the exact schema the model expects.
* Features like `doc_readability` (via `textstat`) and `title_body_divergence` are engineered on-the-fly in milliseconds before passing the batch to `xgb_model.predict()`.



## 11. Infrastructure Friction: OpenMP vs. Asynchronous Event Loops
**The Conflict:** Integrating the XGBoost ranker into the FastAPI (`uvicorn`) pipeline resulted in an immediate Segmentation Fault and Python process death (`Python quit unexpectedly`). The system crash was caused by a threading collision. FastAPI utilizes Python's asynchronous event loop (`asyncio`), while XGBoost's C++ backend utilizes OpenMP for aggressive multi-threading. When the async loop and the C++ threads simultaneously fought for CPU memory allocation on the Apple Silicon architecture, the OS aggressively killed the process to prevent memory corruption.

**The Resolution:** I isolated the memory environments by explicitly neutering the XGBoost multi-threading layer during live inference. By injecting OS-level environment variables (`os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'`) and explicitly setting the XGBoost Booster `nthread` parameter to `1`, the ranker was forced to execute sequentially. This completely resolved the memory collision, allowing sub-15ms live model inference directly inside the FastAPI async loop.

## 12. Phase 5: The Agentic CRAG Synthesizer & Streaming Output
**The Goal:** Prevent hallucinations entirely by implementing a Corrective RAG (CRAG) architecture.
**The Implementation:** 1. **The Evaluator:** Implemented an "Agentic Bouncer" using `llama-3.1-8b-instant`. Before generation occurs, this LLM grades the retrieved context against the user query. If the context is missing the answer, it returns an `Incorrect` flag.
2. **The Web Fallback:** If flagged as `Incorrect`, the system automatically triggers a live web search via the Tavily REST API to pull real-time context from the internet.
3. **The Synthesizer:** Finally, the verified context (either local or web) is passed to a heavy generation model (`llama-3.3-70b-versatile`), which streams the final response back to the client via Server-Sent Events (SSE), complete with inline academic citations.