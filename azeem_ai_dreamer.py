import os
import re
import uuid
import json
import threading
import queue
import time
from datetime import datetime, timedelta, timezone

import requests
import ollama
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from ddgs import DDGS
import html2text
import shutil
from prompt_toolkit import prompt
from rich.markdown import Markdown
from rich.console import Console

console = Console()

# --- COGNITIVE BUFFERS ---
memory_consolidation_queue = queue.Queue()
active_session_rules = []
active_session_facts = []
session_lock = threading.Lock()
last_memory_time = 0.0

# --- EMBEDDING CONFIG (Ollama-served, false for CPU-only) ---
EMBED_MODEL = "qwen3-embedding:4b"
EMBED_DIM = 2560
EMBED_BATCH_SIZE = 16
EMBED_ON_GPU = True           # False for embedder stays on CPU; then GPU is reserved for the chat model
EMBED_KEEP_ALIVE = "30s"

# --- CHAT / CONSOLIDATION CONFIG (single instance, one model) ---
# The selected chat model handles chat, tool calls, keyword extraction, and
# dream consolidation. Same model + same num_ctx + same num_thread on every
# call site means Ollama keeps exactly ONE loaded instance.
#
# At CHAT_NUM_CTX=20000 on 12 GB VRAM, the 14B Q4_K_M will partially offload
# to CPU. If you want 100% GPU, switch to qwen3:14b-q3_k_m at the model
# selection prompt — it fits fully at this context size.
CHAT_NUM_CTX = 20000
CHAT_NUM_THREAD = 14
CHAT_NUM_PREDICT = 14000        # caps generation at ~55K characters (long code files)

# --- TUNABLES ---
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 300
DECAY_INTERVAL_SECONDS = 3600      # scan for stale facts once per hour
DECAY_WINDOW_DAYS = 7              # untouched for this long → demote
DECAY_STEP = 1
SALIENCE_FLOOR = 1
RERANK_FETCH = 10
RERANK_TOP = 3
SALIENCE_WEIGHT = 0.35
CONSOLIDATION_SCROLL_CAP = 150
IDLE_BEFORE_CONSOLIDATION = 305     # seconds of quiet before a dream cycle runs

# Memory tag
MEM_TAG_OPEN = "<memory>"
MEM_TAG_CLOSE = "</memory>"
MEM_TAG_RE = re.compile(
    re.escape(MEM_TAG_OPEN) + r"(.*?)" + re.escape(MEM_TAG_CLOSE),
    re.DOTALL | re.IGNORECASE,
)

# Memory prefix — entries in active_session_facts are stored as "[timestamp] fact"
_MEM_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*")


def _strip_mem_prefix(entry):
    """Remove the leading '[timestamp] ' prefix from a memory entry."""
    return _MEM_PREFIX_RE.sub("", entry, count=1).strip()


def _chat_opts(**extra):
    """Options for the single reasoning model.

    num_ctx and num_thread are pinned to the same values on every call site
    so Ollama reuses one loaded instance across chat, tool calls, keyword
    extraction, and dream consolidation. Sampling params (temperature,
    num_predict, stop) vary per call and do not trigger a reload.
    """
    opts = {"num_ctx": CHAT_NUM_CTX, "num_thread": CHAT_NUM_THREAD}
    opts.update(extra)
    return opts


def _embed_opts():
    """Options for the embedding model — pinned to CPU."""
    opts = {}
    if not EMBED_ON_GPU:
        opts["num_gpu"] = 1
    return opts


def get_user_choice(options):
    while True:
        print("\nAvailable options:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def list_ollama_models():
    try:
        response = ollama.list()
        models_list = response.models if hasattr(response, "models") else response.get("models", [])
        if not models_list:
            print("No models found. Please ensure Ollama is running and models are loaded.")
            return []
        return [m.model for m in models_list]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


# --- DYNAMIC MODEL CONFIG ---
model_names = list_ollama_models()
if model_names:
    selected_model = get_user_choice(model_names)
    print(f"\nYou selected the model: {selected_model}")
    MODEL_NAME = selected_model
else:
    print("No models available to select.")
    MODEL_NAME = "qwen3:14b"

# --- CONFIGURATION ---
MEM_COLLECTION = "fast_local_memories"
RLHF_COLLECTION = "local_rlhf_rules"
DOC_COLLECTION = "document_shards"
width = shutil.get_terminal_size().columns

qdrant = QdrantClient(url="http://localhost:6333")


def get_embedding(text):
    """Embed a single string via Ollama's /api/embed (CPU)."""
    if not text or not text.strip():
        return [0.0] * EMBED_DIM
    try:
        resp = ollama.embed(
            model=EMBED_MODEL,
            input=text,
            options=_embed_opts(),
            keep_alive=EMBED_KEEP_ALIVE,
        )
        embs = resp["embeddings"]
        return embs[0] if embs and isinstance(embs[0], list) else embs
    except Exception as e:
        print(f"[Embed Error] {e}")
        return [0.0] * EMBED_DIM


def get_embeddings_batch(texts, batch_size=EMBED_BATCH_SIZE):
    """Embed a list of strings in batches via Ollama (CPU). Used during ingestion."""
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t and t.strip() else " " for t in texts[i : i + batch_size]]
        try:
            resp = ollama.embed(
                model=EMBED_MODEL,
                input=batch,
                options=_embed_opts(),
                keep_alive=EMBED_KEEP_ALIVE,
            )
            embs = resp["embeddings"]
            if len(embs) != len(batch):
                raise ValueError(f"Expected {len(batch)} embeddings, got {len(embs)}")
            all_embeddings.extend(embs)
        except Exception as e:
            print(f"[Embed Batch Error] {e}")
            all_embeddings.extend([[0.0] * EMBED_DIM for _ in batch])
    return all_embeddings


def warm_up_embedder():
    """Fail fast at startup if the embedding model is missing."""
    try:
        ollama.embed(
            model=EMBED_MODEL,
            input="warmup",
            options=_embed_opts(),
            keep_alive=EMBED_KEEP_ALIVE,
        )
        location = "GPU" if EMBED_ON_GPU else "CPU"
        print(f"[Embedder] '{EMBED_MODEL}' ({EMBED_DIM}-dim) ready on {location}.")
    except Exception as e:
        print(f"[Embedder] ERROR: could not reach '{EMBED_MODEL}'.")
        print(f"  Run: ollama pull {EMBED_MODEL}")
        print(f"  Detail: {e}")


def ensure_collections():
    """Create collections, or migrate them if the vector dimension changed."""
    for col in [MEM_COLLECTION, RLHF_COLLECTION, DOC_COLLECTION]:
        if qdrant.collection_exists(col):
            info = qdrant.get_collection(col)
            current_size = info.config.params.vectors.size
            if current_size != EMBED_DIM:
                print(
                    f"[Migration] Collection '{col}' has vector size {current_size}, "
                    f"expected {EMBED_DIM}. Deleting and recreating (old data will be lost)."
                )
                qdrant.delete_collection(col)
                qdrant.create_collection(
                    collection_name=col,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
        else:
            qdrant.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )


warm_up_embedder()
ensure_collections()


# --- SALIENCE / RETRIEVAL HELPERS ---


def save_fact_async(collection, text, timestamp_str, salience=5):
    """Store a fact with a salience weight and access metadata."""
    try:
        vector = get_embedding(text)
        full_text = f"[{timestamp_str}] {text}"
        deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
        qdrant.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=deterministic_id,
                    vector=vector,
                    payload={
                        "memory": full_text,
                        "raw_fact": text,
                        "timestamp": timestamp_str,
                        "salience": salience,
                        "access_count": 0,
                        "last_accessed": timestamp_str,
                        "superseded": False,
                    },
                )
            ],
        )
        print(f"\n[Stored/Updated] Synapse locked (Salience {salience}/10): '{text}'")
    except Exception as e:
        print(f"\nNeuro-storage error: {str(e)}")


def touch_memories(collection, point_ids):
    """Increment access count, refresh last_accessed, and reinforce salience."""
    if not point_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        recs = qdrant.retrieve(collection_name=collection, ids=point_ids, with_payload=True)
    except Exception:
        return
    for r in recs:
        cur_count = r.payload.get("access_count", 0)
        cur_sal = r.payload.get("salience", 5)
        qdrant.set_payload(
            collection_name=collection,
            payload={
                "access_count": cur_count + 1,
                "last_accessed": now,
                "salience": min(10, cur_sal + 1),
            },
            points=[r.id],
        )


def rerank_by_salience(hits, weight=SALIENCE_WEIGHT):
    """Blend cosine score with normalized salience; drop superseded facts."""
    ranked = []
    for h in hits:
        if h.payload.get("superseded", False):
            continue
        sal = h.payload.get("salience", 5)
        sal_norm = max(0.0, min(1.0, (sal - 1) / 9.0))
        blended = (1.0 - weight) * h.score + weight * sal_norm
        ranked.append((blended, h))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked]


def decay_salience(collection, window_days=DECAY_WINDOW_DAYS, step=DECAY_STEP, floor=SALIENCE_FLOOR):
    """Demote facts not accessed within the window. Runs hourly."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    offset = None
    demoted = 0
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection, limit=128, offset=offset, with_payload=True
        )
        if not points:
            break
        for p in points:
            if p.payload.get("superseded", False):
                continue
            last = p.payload.get("last_accessed")
            if not last:
                continue
            try:
                last_dt = datetime.fromisoformat(last)
            except ValueError:
                continue
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if last_dt >= cutoff:
                continue
            cur = p.payload.get("salience", 5)
            new = max(floor, cur - step)
            if new < cur:
                qdrant.set_payload(
                    collection_name=collection,
                    payload={"salience": new},
                    points=[p.id],
                )
                demoted += 1
        if offset is None:
            break
    if demoted:
        print(f"[Dreamer] Decayed salience on {demoted} stale memories.")


# --- BIOLOGICAL DREAMING CYCLE (BACKGROUND CONSOLIDATION) ---


def deep_sleep_consolidation_daemon(collection, model_name):
    """
    Background thread: waits for a quiet period, then drains the consolidation
    queue, reads high-salience long-term memories, resolves contradictions,
    and re-saves with variable salience. Uses the same loaded instance as chat.

    Raw facts that get superseded by consolidated versions are removed from
    active_session_facts so the next turn's system prompt doesn't contain both
    the raw and refined copy of the same memory. The comparison strips the
    '[timestamp] ' prefix from each entry and does an exact-match against the
    superseded text — this prevents a short raw fact (e.g. "Azeem") from
    stripping an unrelated longer entry that merely contains that substring.
    """
    print(f"[Dreamer] Consolidation daemon online (model='{model_name}', shared instance).")
    last_decay = time.time()

    while True:
        time.sleep(15)

        # Hourly salience decay (no LLM call — runs even during activity)
        if time.time() - last_decay > DECAY_INTERVAL_SECONDS:
            try:
                decay_salience(collection)
            except Exception as e:
                print(f"[Dreamer] Decay error: {e}")
            last_decay = time.time()

        # Only consolidate after a quiet period — no new memories arriving
        if (time.time() - last_memory_time) < IDLE_BEFORE_CONSOLIDATION:
            continue

        # Drain queue without blocking
        batch = []
        while True:
            try:
                batch.append(memory_consolidation_queue.get_nowait())
            except queue.Empty:
                break
        if not batch:
            continue

        print("\n[Cerebral Dream Cycle] Consolidating short-term memory registers...")

        try:
            existing_points = []
            offset = None
            while True:
                points, offset = qdrant.scroll(
                    collection_name=collection, limit=100, offset=offset, with_payload=True
                )
                if not points:
                    break
                existing_points.extend(points)
                if offset is None or len(existing_points) >= 500:
                    break

            existing_points = [p for p in existing_points if not p.payload.get("superseded", False)]
            existing_points.sort(key=lambda p: p.payload.get("salience", 5), reverse=True)
            existing_points = existing_points[:CONSOLIDATION_SCROLL_CAP]

            if existing_points:
                existing_memories = "\n".join(
                    [
                        f"[sal={p.payload.get('salience', 5)} ts={p.payload.get('timestamp', '?')}] "
                        f"{p.payload.get('raw_fact', '')}"
                        for p in existing_points
                    ]
                )
            else:
                existing_memories = "None yet."

            recent_text = "\n".join(f"NEW: {t}" for t in batch)

            consolidation_prompt = (
                "You are the sub-conscious memory consolidation engine of an advanced brain matrix.\n"
                "Review these newly acquired short-term observations alongside existing long-term facts.\n"
                "Your job is to resolve discrepancies, combine fractional updates into unified clean concepts,\n"
                "and assign a salience score to each result.\n\n"
                "Return ONLY a JSON array of objects with this exact shape:\n"
                '[\n'
                '  {"fact": "the consolidated fact text", "salience": 1-10, "supersedes": ["exact raw_fact from EXISTING, or empty list"]}\n'
                ']\n\n'
                "Salience rules:\n"
                "- 8-10: verified by multiple independent sources, or explicitly reinforced by the user.\n"
                "- 5-7: standalone observation, plausible but unverified.\n"
                "- 1-3: fragmentary, ambiguous, or superseded by a newer fact.\n"
                "- If a new fact replaces an older one, put the older fact's exact raw text in \"supersedes\".\n\n"
                f"EXISTING MEMORIES:\n{existing_memories}\n\n"
                f"RECENT RAW MATERIAL:\n{recent_text}\n"
            )

            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "system", "content": consolidation_prompt}],
                format="json",
                options=_chat_opts(temperature=0.2),
            )
            raw_json = resp["message"]["content"].strip()
            raw_json = re.sub(r"^```json|```$", "", raw_json, flags=re.IGNORECASE).strip()

            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
            if not isinstance(parsed, list):
                print("[Dreamer] Consolidation returned non-list; skipping cycle.")
                continue

            current_time = datetime.now(timezone.utc).isoformat()
            saved = 0
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                fact = (item.get("fact") or "").strip()
                if not fact:
                    continue
                sal = item.get("salience", 7)
                try:
                    sal = int(sal)
                except (TypeError, ValueError):
                    sal = 7
                sal = max(1, min(10, sal))

                # Collect exact raw texts this consolidated fact supersedes
                superseded_texts = []
                for old in item.get("supersedes") or []:
                    old_clean = (old or "").strip()
                    if not old_clean:
                        continue
                    superseded_texts.append(old_clean)
                    try:
                        qdrant.set_payload(
                            collection_name=collection,
                            payload={"salience": 2, "superseded": True},
                            points=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="raw_fact",
                                        match=models.MatchValue(value=old_clean),
                                    )
                                ]
                            ),
                        )
                    except Exception:
                        pass

                save_fact_async(collection, fact, current_time, salience=sal)

                # Deduplicate active_session_facts — exact match on the fact
                # portion after stripping the '[timestamp] ' prefix. This avoids
                # accidentally removing unrelated entries that happen to contain
                # the superseded text as a substring.
                with session_lock:
                    if superseded_texts:
                        superseded_set = {t.strip() for t in superseded_texts}
                        active_session_facts[:] = [
                            entry
                            for entry in active_session_facts
                            if _strip_mem_prefix(entry) not in superseded_set
                        ]
                    if f"[{current_time}] {fact}" not in active_session_facts:
                        active_session_facts.append(f"[{current_time}] {fact}")
                saved += 1

            if saved:
                print(f"[Dream Cycle Complete] {saved} facts consolidated.")

        except Exception as e:
            print(f"[Dream Cycle Interrupted] Processing exception caught: {e}")


# Spawn the subconscious worker thread — shares the chat model instance
threading.Thread(
    target=deep_sleep_consolidation_daemon,
    args=(MEM_COLLECTION, MODEL_NAME),
    daemon=True,
).start()


# --- DOCUMENT INGESTION ---


def split_into_chunks(text, chunk_chars=CHUNK_CHARS, overlap_chars=CHUNK_OVERLAP):
    """Character-based sliding window with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap_chars
    return chunks


def ingest_large_document(file_path):
    """Ingest a document without destroying previously ingested documents."""
    expanded_path = os.path.expanduser(file_path.strip())
    print(f"\n[Ingestion Pipeline] Slicing document at: '{expanded_path}'...")

    if not os.path.exists(expanded_path):
        return f"File system error: Target path '{expanded_path}' does not exist."

    try:
        with open(expanded_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
    except Exception as e:
        return f"Document slicing operation aborted: {str(e)}"

    if not raw_text.strip():
        return "Document is empty; nothing ingested."

    chunks = split_into_chunks(raw_text)
    source = os.path.basename(expanded_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f" Document parsed. {len(chunks)} chunks from '{source}'. Embedding in batches...")

    vectors = get_embeddings_batch(chunks)
    if len(vectors) != len(chunks):
        return f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks."

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}:{i}:{chunk[:64]}"))
        points.append(
            models.PointStruct(
                id=pid,
                vector=vec,
                payload={
                    "text": chunk,
                    "source": source,
                    "source_path": expanded_path,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "timestamp": timestamp,
                },
            )
        )

    for i in range(0, len(points), 64):
        qdrant.upsert(collection_name=DOC_COLLECTION, points=points[i : i + 64])

    return f"Ingested {len(points)} chunks from '{source}'."


def query_document_shards(user_query, source_filter=None, top_k=5):
    """Retrieve relevant document chunks across all ingested sources."""
    try:
        query_vector = get_embedding(user_query)
        query_filter = None
        if source_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter),
                    )
                ]
            )
        search_result = qdrant.query_points(
            collection_name=DOC_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        if not search_result or not search_result.points:
            return ""
        retrieved_blocks = []
        print("\n[Semantic Match] Ingesting relevant document lines into context:")
        for point in search_result.points:
            src = point.payload.get("source", "doc")
            idx = point.payload.get("chunk_index", "?")
            total = point.payload.get("chunk_total", "?")
            print(f" -> Extracted Shard Segment: '{src}' chunk {idx}/{total}")
            retrieved_blocks.append(
                f"--- Ingested Document Excerpt ({src} chunk {idx}/{total}) ---\n"
                f"{point.payload.get('text', '')}\n"
            )
        return "\n".join(retrieved_blocks)
    except Exception as e:
        return f"Document shard retrieval failure: {str(e)}"


def list_ingested_documents():
    """Return sorted distinct source filenames currently in DOC_COLLECTION."""
    sources = set()
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=DOC_COLLECTION, limit=256, offset=offset, with_payload=True
        )
        if not points:
            break
        for p in points:
            sources.add(p.payload.get("source", "?"))
        if offset is None:
            break
    return sorted(sources)


# --- WEB TOOLS ---


def scrape_w3m_style(url):
    if not url or not url.startswith("http"):
        return "Invalid or empty source URL target."
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/155.0.8059.79 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=4)
        response.raise_for_status()
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.ignore_emphasis = True
        converter.body_width = 0
        return "\n".join(
            [line.strip() for line in converter.handle(response.text).splitlines() if line.strip()]
        )
    except Exception as e:
        return f"CLI Text download dropped: {str(e)}"


def run_parallel_hybrid_search(query):
    try:
        with DDGS() as ddgs:
            text_hits = list(ddgs.text(query, max_results=4))
        blended_payloads = []
        for item in text_hits:
            if isinstance(item, dict):
                blended_payloads.append(item)
            else:
                blended_payloads.append(
                    {
                        "title": getattr(item, "title", "Web Result"),
                        "url": getattr(item, "url", getattr(item, "href", "")),
                        "body": getattr(item, "body", ""),
                    }
                )
        return blended_payloads
    except Exception as e:
        print(f"Search infrastructure exception caught: {str(e)}")
        return []


def process_tool_calls(turn_messages, agent_tools_schema, MODEL_NAME):
    loop_count = 0
    while loop_count < 5:
        loop_count += 1
        response = ollama.chat(
            model=MODEL_NAME,
            messages=turn_messages,
            tools=agent_tools_schema,
            options=_chat_opts(),   # pinned to match the streaming call
        )
        tool_calls = response.get("message", {}).get("tool_calls", [])
        if not tool_calls:
            break
        calls_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
        turn_messages.append(response["message"])
        for tool_call in calls_list:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            if function_name == "execute_new_web_search":
                refined_hits = run_parallel_hybrid_search(arguments["query"])
                raw_result = "\n".join(
                    [
                        f"Title: {h.get('title')}\nURL: {h.get('url') or h.get('href')}\n"
                        f"Summary: {h.get('body')}\n---"
                        for h in refined_hits[:3]
                    ]
                )
                tool_result = f"\n\n{raw_result}\n\n"
                print(f" [Agent Search] AI executed web lookup for: '{arguments['query']}'")
            elif function_name == "click_and_read_link":
                raw_scrape = scrape_w3m_style(arguments["url"])
                tool_result = f"\n\n{raw_scrape}\n\n"
                print(f" [Agent Scrape] AI read website text from: {arguments['url']}")
            else:
                tool_result = "Target execution mismatch."
            turn_messages.append({"role": "tool", "name": function_name, "content": tool_result})
    return turn_messages


agent_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "execute_new_web_search",
            "description": "Search the live internet using an explicit keyword string to fetch real-time updates or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The explicit 3-to-4 word search engine keyword string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_and_read_link",
            "description": "Read the full structural text content from a specific target URL link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The absolute URL to read."}
                },
                "required": ["url"],
            },
        },
    },
]


# --- MAIN SESSION ---


def run_session():
    global active_session_rules, active_session_facts, last_memory_time

    m_points, _ = qdrant.scroll(collection_name=MEM_COLLECTION, limit=100, with_payload=True)
    r_points, _ = qdrant.scroll(collection_name=RLHF_COLLECTION, limit=100, with_payload=True)

    with session_lock:
        active_session_facts = [p.payload["memory"] for p in m_points if p.payload]
        active_session_rules = [p.payload["memory"] for p in r_points if p.payload]

    # Start the idle clock so the daemon doesn't fire immediately on launch
    last_memory_time = time.time()

    chat_history = []
    print("=" * 80)
    print(f" Neural Framework Loaded. Found {len(active_session_facts)} consolidated cortex facts.")
    print(f" Loaded {len(active_session_rules)} active alignment filters.")
    print(f" Active model:  {MODEL_NAME}  (GPU, num_ctx={CHAT_NUM_CTX}, single instance)")
    print(f" Embedding:     {EMBED_MODEL}  ({EMBED_DIM}-dim, "
          f"{'GPU' if EMBED_ON_GPU else 'CPU'})")
    print(f" Long output:   num_predict={CHAT_NUM_PREDICT}  (~{CHAT_NUM_PREDICT * 4} chars)")
    print(f" Dream cycle:   fires after {IDLE_BEFORE_CONSOLIDATION}s of no new memories")
    print(" UI Toggles per turn:")
    print("  -> Ingest new documents with 'ingest path/to/file.md'")
    print("  -> List ingested documents with 'ls docs'")
    print("  -> Read a url with 'read http://www.example.com'")
    print("  -> Add '--doc' to include local document shards in the answer.")
    print("  -> Add '--web' to execute real-time internet lookups.")
    print("  -> Multiline inputs: press Left ALT + Enter to submit.")
    print("  -> CTRL + C to end the session.")
    print("=" * 80)

    while True:
        try:
            with session_lock:
                memory_context = (
                    "\n".join(f"- {f}" for f in active_session_facts)
                    if active_session_facts
                    else "None"
                )
                alignment_context = (
                    "\n".join(f"- {r}" for r in active_session_rules)
                    if active_session_rules
                    else "None"
                )

            user_raw_input = prompt("\n You: \n", multiline=True).strip()
            if not user_raw_input:
                continue

            if user_raw_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            now = datetime.now()
            current_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            current_day_name = now.strftime("%A")
            current_year = now.strftime("%Y")

            # Word-boundary match so --docs isn't treated as --doc
            force_doc = bool(re.search(r"(?<!\S)--doc(?!\S)", user_raw_input))
            force_web = bool(re.search(r"(?<!\S)--web(?!\S)", user_raw_input))
            user_input = re.sub(r"(?<!\S)--doc(?!\S)", "", user_raw_input)
            user_input = re.sub(r"(?<!\S)--web(?!\S)", "", user_input).strip()

            # --docs shortcut
            if user_input.strip() == "ls docs":
                docs = list_ingested_documents()
                if docs:
                    print("Ingested documents:")
                    for d in docs:
                        print(f"  - {d}")
                else:
                    print("No documents ingested yet.")
                continue

            local_doc_context = ""
            web_search_context = ""
            read_context = ""

            if user_raw_input.lower().startswith("ingest "):
                target_doc_path = user_raw_input[7:].strip()
                status_msg = ingest_large_document(target_doc_path)
                print(f" System Ingest Status: {status_msg}")
                continue

            if user_raw_input.lower().startswith("rlhf "):
                if not chat_history:
                    print("No conversation history to correct yet.")
                    continue
                print("\n[RLHF Mode] What did the model do wrong? Enter structural feedback:")
                feedback = input(" Feedback: ").strip()
                if feedback:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    formatted_rule = f"User Preference: {feedback}"
                    with session_lock:
                        active_session_rules.append(f"[{current_time}] {formatted_rule}")
                    save_fact_async(RLHF_COLLECTION, formatted_rule, current_time, salience=10)
                    print("Internal alignment rules adjusted instantly.")
                continue

            if user_raw_input.lower().startswith("read "):
                target_url_path = user_raw_input[5:].strip()
                readingmaterial = scrape_w3m_style(target_url_path)
                read_context = (
                    f"\n[Website Source]\nURL: {target_url_path}\n"
                    f"Website Text:\n {readingmaterial}\n\n---\n"
                )

            if force_doc:
                try:
                    collection_info = qdrant.get_collection(DOC_COLLECTION)
                    has_shards = collection_info.points_count > 0
                except Exception:
                    has_shards = False
                if has_shards:
                    retrieved_chunks = query_document_shards(user_input)
                    if retrieved_chunks.strip():
                        local_doc_context = (
                            f"\n[INGESTED LOCAL DOCUMENT CONTENT SHARDS]:\n\n"
                            f"{retrieved_chunks}\n\n"
                        )
                else:
                    print(
                        "Warning: '--doc' requested but no active document shards exist. "
                        "Run 'ingest' first."
                    )

            if force_web:
                print(" AI is formulating optimized search keywords...")
                keyword_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise keyword extraction engine. Output ONLY a clean "
                            "3-to-4 word search query based on the text. Never include explanations, "
                            "markdown tags, introduction phrases, or reasoning blocks."
                        ),
                    },
                    {"role": "user", "content": f"Extract query for: {user_input}"},
                ]
                kw_resp = ollama.chat(
                    model=MODEL_NAME,   # same loaded instance as chat
                    messages=keyword_messages,
                    options=_chat_opts(num_predict=16, temperature=0.0, stop=["\n"]),
                )
                clean_query = kw_resp["message"]["content"].strip().replace('"', "").replace("'", "")
                clean_query = re.sub(
                    r"^(Topic:|Keywords:|Query:)\s*", "", clean_query, flags=re.IGNORECASE
                )
                if not clean_query or len(clean_query) < 2:
                    clean_query = user_input
                print(f" [Search] Scouring internet sources for: '{clean_query}'...")
                initial_hits = run_parallel_hybrid_search(clean_query)
                if initial_hits:
                    web_search_context = (
                        f"\n[INITIAL AVAILABLE WEB SOURCES FOR KEYWORDS: {clean_query}]:\n"
                    )
                    for i, r in enumerate(initial_hits[:5], 1):
                        link = r.get("url") or r.get("href", "")
                        body_desc = r.get("body") or r.get("summary", "")
                        web_search_context += (
                            f"\n\n\nSource [{i}]: {r.get('title')}\n"
                            f"URL: {link}\nWebsite Text or Summary: {body_desc}\n---\n"
                        )

            combined_web_context = local_doc_context + web_search_context

            system_prompt = (
                f"You are an elite research AI running on {MODEL_NAME}.\n"
                f"Your active current time is: {current_day_name}, {current_timestamp} "
                f"(Year: {current_year}).\n"
                f"AGENT OPERATION DIRECTIVE:\n"
                f"- Review the supplied initial context frames carefully.\n"
                f"- If specific details are missing or blocked by placeholders, execute tool calls autonomously to read full text via `click_and_read_link` or re-search via `execute_new_web_search`.\n"
                f"- Keep looping and gathering data until you have the exact facts required to completely satisfy the user's prompt.\n\n"
                f"COGNITIVE RESONANCE & TRUTH SEEKING:\n"
                f"- When faced with multiple sources claiming conflicting truths, research more and acknowledge lack of info. Only present information verified by multiple credible sources.\n\n"
                f"CITE YOUR SOURCES:\n"
                f"- As a fact driven AI you are open minded and willing to investigate any topic with scientific and academic rigour. Cite all of your sources for facts and claims. Provide a link to every source you read or that was relevant in the response.\n\n"
                f"[HUMAN FEEDBACK RULES]:\n{alignment_context}\n\n"
                f"[CURRENT FACTS KNOWN ABOUT THE USER]:\n{memory_context}\n\n"
                f"INLINE EXTRACTION GUIDELINES:\n"
                f"ONLY IF the user explicitly shares a new personal detail missing from the facts list, OR explicitly instructs you to remember/save info, THEN append the information to memory using this EXACT format at the very end of your response:\n"
                f"{MEM_TAG_OPEN}The user's [property] is [value]{MEM_TAG_CLOSE}\n"
                f"CRITICAL CONSTRAINT: If the user did NOT share a new personal fact, task, or rule in this turn, DO NOT output any {MEM_TAG_OPEN} tags whatsoever. Never output generic templates like 'The user's [property] is [value]'.\n\n"
                f"CURRENT READING, WEB, AND DOCUMENT CONTEXT:\n"
                f"{read_context}\n\n"
                f"{combined_web_context}\n\n"
            )

            timed_user_input = f"[{current_timestamp}] {user_input}"
            turn_messages = (
                [{"role": "system", "content": system_prompt}]
                + chat_history
                + [{"role": "user", "content": timed_user_input}]
            )
            turn_messages = process_tool_calls(
                turn_messages=turn_messages,
                agent_tools_schema=agent_tools_schema,
                MODEL_NAME=MODEL_NAME,
            )

            # Salience-weighted retrieval
            user_embedding = get_embedding(user_input)
            raw_hits = qdrant.query_points(
                collection_name=MEM_COLLECTION, query=user_embedding, limit=RERANK_FETCH
            ).points
            ranked = rerank_by_salience(raw_hits, weight=SALIENCE_WEIGHT)
            top_hits = ranked[:RERANK_TOP]
            touch_memories(MEM_COLLECTION, [h.id for h in top_hits])

            print(" AI: \n", end="", flush=True)
            full_response = ""
            thoughts = ""
            output = ""
            stream = ollama.chat(
                model=MODEL_NAME,
                messages=turn_messages,
                stream=True,
                options=_chat_opts(num_predict=CHAT_NUM_PREDICT),
            )
            in_thinking = False
            for chunk in stream:
                msg = getattr(chunk, "message", None)
                if msg:
                    thinking = getattr(msg, "thinking", "")
                    if thinking:
                        if not in_thinking:
                            print("\n", end="", flush=True)
                            full_response += "\n\n"
                            thoughts += "\n\n"
                            in_thinking = True
                        print(thinking, end="", flush=True)
                        full_response += thinking
                        thoughts += thinking
                        continue
                    content = getattr(msg, "content", "")
                    if content:
                        if in_thinking:
                            print("\n\n\n", end="", flush=True)
                            full_response += "\n\n\n"
                            output += "\n\n\n"
                            in_thinking = False
                        print(content, end="", flush=True)
                        full_response += content
                        output += content
            print()
            print("\n")
            horizontal_rule = "-" * width
            print(horizontal_rule)
            console.print(Markdown("### Thinking"))
            console.print(Markdown(f"```{thoughts}```"))
            console.print(Markdown("### Response"))
            console.print(Markdown(output))

            # --- AMYGDALA RECONSOLIDATION ROUTER ---
            memory_matches = MEM_TAG_RE.findall(full_response)
            clean_for_history = MEM_TAG_RE.sub("", full_response).strip()
            for fact in memory_matches:
                fact_clean = fact.strip()
                if (
                    not fact_clean
                    or "[property]" in fact_clean.lower()
                    or "[value]" in fact_clean.lower()
                    or "the user's is" in fact_clean.lower()
                ):
                    continue
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                memory_consolidation_queue.put(fact_clean)
                last_memory_time = time.time()
                with session_lock:
                    active_session_facts.append(f"[{current_time}] {fact_clean}")
                print(
                    f" [Synaptic Working Buffer Stage]: Sent '{fact_clean}' to Subconscious Queue."
                )

            chat_history.append({"role": "user", "content": timed_user_input})
            chat_history.append({"role": "assistant", "content": clean_for_history})

        except KeyboardInterrupt:
            print(
                "\nSession closing... type 'deactivate' to exit the venv and 'cd' to take you home. "
                "or just type 'mm' if you have Azeem's bash shortcuts installed."
            )
            break


if __name__ == "__main__":
    run_session()