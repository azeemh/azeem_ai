import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import uuid
import json
import threading
import queue
import time
from datetime import datetime
import requests
import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import html2text
import shutil
from prompt_toolkit import prompt

# Rich markdown decoration
from rich.markdown import Markdown
from rich.console import Console
console = Console()

# Global Cognitive Buffers for Real-Time Synchronization
memory_consolidation_queue = queue.Queue()
active_session_rules = []
active_session_facts = []

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
        models_list = response.models if hasattr(response, 'models') else response.get('models', [])
        if not models_list:
            print("No models found. Please ensure Ollama is running and models are loaded.")
            return []
        return [m.model for m in models_list]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

# --- DYNAMIC MODEL CONFIG -- #
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

print("🚀 Loading instant 384-dim embedding model on CPU (Offline Mode)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

qdrant = QdrantClient(url="http://localhost:6333")

# Initialize collections using Cosine distance metrics
for col in [MEM_COLLECTION, RLHF_COLLECTION, DOC_COLLECTION]:
    if not qdrant.collection_exists(col):
        from qdrant_client.models import Distance, VectorParams
        qdrant.create_collection(
            collection_name=col, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def get_embedding(text):
    return embed_model.encode(text).tolist()

def extract_and_execute_tool(model_output):
    tool_call_match = re.search(r'\{"name":\s*"execute_new_web_search".*?\}', model_output, re.DOTALL)
    if tool_call_match:
        try:
            tool_call_json = json.loads(tool_call_match.group(0))
            query = tool_call_json["arguments"]["query"]
            print(f"Intercepted Tool Call! Querying: {query}")
        except json.JSONDecodeError:
            print("Found tool call but failed to decode JSON.")
    return None

def save_fact_async(collection, text, timestamp_str, salience=5):
    """Upgraded with a biological 'Salience Vector Weight' saved as payload metadata."""
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
                        "access_count": 1,
                        "last_accessed": timestamp_str
                    }
                )
            ]
        )
        print(f"\n💾 [Stored/Updated] Neural Synapse locked (Salience Level: {salience}/10): '{text}'")
    except Exception as e:
        print(f"\n❌ Neuro-storage error: {str(e)}")

# --- BIOLOGICAL DREAMING CYCLE LAYER (BACKGROUND CONSOLIDATION) ---
def deep_sleep_consolidation_daemon(collection, model_name):
    """
    Background Thread: Mimics the biological sleep cycle by picking up live context memories,
    checking for contradictions, dropping expired facts, and consolidating micro-insights into schemas.
    """
    while True:
        time.sleep(15)  # Scans internal consolidation registers every 15 seconds
        if memory_consolidation_queue.empty():
            continue

        print("\n🌙 [Cerebral Dream Cycle] Consolidating short-term memory registers...")
        raw_materials = []
        while not memory_consolidation_queue.empty():
            raw_materials.append(memory_consolidation_queue.get())

        # Pull down current base long-term memories from Qdrant to find semantic overlap
        try:
            points, _ = qdrant.scroll(collection_name=collection, limit=100, with_payload=True)
            existing_memories = [p.payload["raw_fact"] for p in points if p.payload and "raw_fact" in p.payload]
        except Exception:
            existing_memories = []

        consolidation_prompt = (
            "You are the sub-conscious memory consolidation engine of an advanced brain matrix.\n"
            "Review these newly acquired short-term observations alongside existing long-term facts.\n"
            "Your job is to resolve discrepancies, combine fractional updates into unified clean concepts, and drop fluff.\n\n"
            f"[Existing Long-Term Architecture]:\n{json.dumps(existing_memories)}\n\n"
            f"[New Unprocessed Input Influx]:\n{json.dumps(raw_materials)}\n\n"
            "Output ONLY a clean, valid JSON list of optimized strings representing the new truth boundaries to update.\n"
            "Example format: [\"The user prefers dark mode for text outputs.\", \"The user lives in London.\"]"
        )

        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "system", "content": consolidation_prompt}],
                options={"temperature": 0.2}
            )
            raw_json = resp['message']['content'].strip()
            # Clean up wrap highlights if model hallucinates markdown containers
            raw_json = re.sub(r'^```json|```$', '', raw_json, flags=re.IGNORECASE).strip()
            
            refined_facts = json.loads(raw_json)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for fact in refined_facts:
                # Re-save cleaner, condensed semantic representations back to long-term memory
                save_fact_async(collection, fact, current_time, salience=7)
                if fact not in active_session_facts:
                    active_session_facts.append(f"[{current_time}] {fact}")
                    
            print("✨ [Dream Cycle Complete] Synapses pruned and structured successfully.")
        except Exception as e:
            print(f"⚠️ [Dream Cycle Interrupted] Processing exception caught: {e}")

# Spawn the subconscious worker thread
threading.Thread(target=deep_sleep_consolidation_daemon, args=(MEM_COLLECTION, MODEL_NAME), daemon=True).start()

def ingest_large_document(file_path):
    expanded_path = os.path.expanduser(file_path.strip())
    print(f"\n⚡ [Ingestion Pipeline] Slicing large document at: '{expanded_path}'...")
    if not os.path.exists(expanded_path):
        return f"File system error: Target path '{expanded_path}' does not exist."
    try:
        with open(expanded_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        total_lines = len(lines)
        print(f"📊 Document parsed. Total lines detected: {total_lines}")
        window_size, overlap, chunk_count = 500, 100, 0
        qdrant.delete_collection(DOC_COLLECTION)
        qdrant.create_collection(collection_name=DOC_COLLECTION, vectors_config={"size": 384, "distance": "Cosine"})
        start_idx = 0
        while start_idx < total_lines:
            end_idx = min(start_idx + window_size, total_lines)
            chunk_text = "".join(lines[start_idx:end_idx]).strip()
            if chunk_text:
                chunk_count += 1
                vector = get_embedding(chunk_text[:1500])
                qdrant.upload_collection(
                    collection_name=DOC_COLLECTION, vectors=[vector],
                    payload=[{"text": chunk_text, "lines": f"{start_idx}-{end_idx}", "source": expanded_path}]
                )
            start_idx += (window_size - overlap)
            if end_idx == total_lines: break
        return f"Ingested {total_lines} lines across {chunk_count} distinct sliding-window partitions."
    except Exception as e:
        return f"Document slicing operation aborted: {str(e)}"

def query_document_shards(user_query):
    try:
        query_vector = get_embedding(user_query)
        search_result = qdrant.query_points(collection_name=DOC_COLLECTION, query=query_vector, limit=3, with_payload=True)
        if not search_result or not search_result.points: return ""
        retrieved_blocks = []
        print("\n🎯 [Semantic Match] Ingesting relevant document lines into context:")
        for point in search_result.points:
            lines_range = point.payload.get('lines', 'Unknown')
            src = os.path.basename(point.payload.get('source', 'doc'))
            print(f"  -> Extracted Shard Segment: '{src}' Lines: {lines_range}")
            retrieved_blocks.append(f"--- Ingested Document Excerpt (Lines {lines_range}) ---\n{point.payload.get('text', '')}\n")
        return "\n".join(retrieved_blocks)
    except Exception as e: return f"Document shard retrieval failure: {str(e)}"

def scrape_w3m_style(url):
    if not url or not url.startswith("http"): return "Invalid or empty source URL target."
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.8059.79 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=4)
        response.raise_for_status()
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.ignore_emphasis = True
        converter.body_width = 0
        return "\n".join([line.strip() for line in converter.handle(response.text).splitlines() if line.strip()])
    except Exception as e: return f"CLI Text download dropped: {str(e)}"

def run_parallel_hybrid_search(query):
    try:
        with DDGS() as ddgs:
            text_hits = list(ddgs.text(query, max_results=4))
        blended_payloads = []
        for item in text_hits:
            if isinstance(item, dict):
                blended_payloads.append(item)
            else:
                blended_payloads.append({
                    'title': getattr(item, 'title', 'Web Result'),
                    'url': getattr(item, 'url', getattr(item, 'href', '')),
                    'body': getattr(item, 'body', '')
                })
        return blended_payloads
    except Exception as e:
        print(f"⚠️  Search infrastructure exception caught: {str(e)}")
        return []

def process_tool_calls(turn_messages, agent_tools_schema, MODEL_NAME):
    loop_count = 0
    while loop_count < 5:
        loop_count += 1
        response = ollama.chat(model=MODEL_NAME, messages=turn_messages, tools=agent_tools_schema)
        tool_calls = response.get('message', {}).get('tool_calls', [])

        if not tool_calls:
            break

        calls_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
        turn_messages.append(response['message'])

        for tool_call in calls_list:
            function_name = tool_call['function']['name']
            arguments = tool_call['function']['arguments']

            if function_name == 'execute_new_web_search':
                refined_hits = run_parallel_hybrid_search(arguments['query'])
                raw_result = "\n".join([f"Title: {h.get('title')}\nURL: {h.get('url') or h.get('href')}\nSummary: {h.get('body')}\n---" for h in refined_hits[:3]])
                tool_result = f"\n<web_search_payload>\n{raw_result}\n</web_search_payload>\n"
                print(f"🌐 [Agent Search] AI executed web lookup for: '{arguments['query']}'")

            elif function_name == 'click_and_read_link':
                raw_scrape = scrape_w3m_style(arguments['url'])
                tool_result = f"\n<web_scrape_payload>\n{raw_scrape}\n</web_scrape_payload>\n"
                print(f"📄 [Agent Scrape] AI read website text from: {arguments['url']}")
            else:
                tool_result = "Target execution mismatch."

            turn_messages.append({'role': 'tool', 'name': function_name, 'content': tool_result})
    return turn_messages

agent_tools_schema = [
    {
        'type': 'function',
        'function': {
            'name': 'execute_new_web_search',
            'description': 'Search the live internet using an explicit keyword string to fetch real-time updates or news.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string', 'description': 'The explicit 3-to-4 word search engine keyword string.'}},
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'click_and_read_link',
            'description': 'Read the full structural text content from a specific target URL link.',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string', 'description': 'The absolute URL to read.'}},
                'required': ['url']
            }
        }
    }
]

def run_session():
    global active_session_rules, active_session_facts
    m_points, _ = qdrant.scroll(collection_name=MEM_COLLECTION, limit=100, with_payload=True)
    r_points, _ = qdrant.scroll(collection_name=RLHF_COLLECTION, limit=100, with_payload=True)
    
    active_session_facts = [p.payload["memory"] for p in m_points if p.payload]
    active_session_rules = [p.payload["memory"] for r in r_points if r.payload]
    chat_history = []

    print("=" * 80)
    print(f"🧠 Neural Framework Loaded. Found {len(active_session_facts)} consolidated cortex facts.")
    print(f"🎯 Loaded {len(active_session_rules)} active alignment filters.")
    print(f"⚡ Core Model Active: {MODEL_NAME}")
    print("UI Toggles per turn:\n  -> Ingest new documents with 'ingest path/to/file.md'\n  -> Read a url with 'read http://www.exampleurl.com' \n  -> Add '--doc' to tell AI to read local document vector shards & incorporate the documents into it's answer.\n  -> Add '--web' to execute real-time internet lookups.")
    print("  -> Multiline inputs are now supported so you have to press Left ALT + Enter to submit your input.")
    print("  -> Use CTRL + C to end the session.")
    print("=" * 80)

    while True:
        try:
            has_shards = False
            memory_context = "\n".join(f"- {f}" for f in active_session_facts) if active_session_facts else "None"
            alignment_context = "\n".join(f"- {r}" for r in active_session_rules) if active_session_rules else "None"
            
            user_raw_input = prompt("\n👤 You: \n", multiline=True).strip()
            if not user_raw_input:
                continue
            if user_raw_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            now = datetime.now()
            current_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            current_day_name = now.strftime("%A")
            current_year = now.strftime("%Y")

            force_doc = "--doc" in user_raw_input
            force_web = "--web" in user_raw_input
            user_input = user_raw_input.replace("--doc", "").replace("--web", "").strip()
            local_doc_context = ""
            web_search_context = ""
            read_context = ""

            if user_raw_input.lower().startswith("ingest "):
                target_doc_path = user_raw_input[7:].strip()
                status_msg = ingest_large_document(target_doc_path)
                print(f"📢 System Ingest Status: {status_msg}")
                continue

            if user_raw_input.lower().startswith("rlhf "):
                if not chat_history:
                    print("❌ No conversation history to correct yet.")
                    continue

                print("\n🚨 [RLHF Mode] What did the model do wrong? Enter structural feedback:")
                feedback = input("👉 Feedback: ").strip()
                if feedback:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    formatted_rule = f"User Preference: {feedback}"
                    active_session_rules.append(f"[{current_time}] {formatted_rule}")
                    save_fact_async(RLHF_COLLECTION, formatted_rule, current_time, salience=10)
                    print("✅ Internal alignment rules adjusted instantly.")
                continue

            if user_raw_input.lower().startswith("read "):
                target_url_path = user_raw_input[5:].strip()
                readingmaterial = scrape_w3m_style(target_url_path)
                read_context = f"\n[Website Source]\nURL: {target_url_path}\nWebsite Text:\n {readingmaterial}\n\n---\n"

            if force_doc:
                try:
                    collection_info = qdrant.get_collection(DOC_COLLECTION)
                    has_shards = collection_info.points_count > 0
                except Exception: has_shards = False
                if has_shards:
                    retrieved_chunks = query_document_shards(user_input)
                    if retrieved_chunks.strip():
                        local_doc_context = f"\n[INGESTED LOCAL DOCUMENT CONTENT SHARDS]:\n<document>\n{retrieved_chunks}\n</document>\n"
                else:
                    print("⚠️  Warning: '--doc' requested but no active document shards exist on disk. Run 'ingest' first.")

            if force_web:
                print("🧠 AI is formulating optimized search keywords...")
                keyword_messages = [
                    {"role": "system", "content": "You are a precise keyword extraction engine. Output ONLY a clean 3-to-4 word search query based on the text. Never include explanations, markdown tags, introduction phrases, or reasoning blocks."},
                    {"role": "user", "content": f"Extract query for: {user_input}"}
                ]
                kw_resp = ollama.chat(model=MODEL_NAME, messages=keyword_messages, options={"num_predict": 16, "temperature": 0.0, "stop": ["<think>", "\n"]})
                clean_query = kw_resp['message']['content'].strip().replace('"', '').replace("'", "")
                clean_query = re.sub(r'^(Topic:|Keywords:|Query:)\s*', '', clean_query, flags=re.IGNORECASE)
                if not clean_query or len(clean_query) < 2: clean_query = user_input

                print(f"🔍 [Search] Scouring internet sources for: '{clean_query}'...")
                initial_hits = run_parallel_hybrid_search(clean_query)
                if initial_hits:
                    web_search_context = f"\n[INITIAL AVAILABLE WEB SOURCES FOR KEYWORDS: {clean_query}]:\n"
                    for i, r in enumerate(initial_hits[:5], 1):
                        link = r.get('url') or r.get('href', '')
                        body_desc = r.get('body') or r.get('summary', '')
                        web_search_context += f"\n\n\nSource [{i}]: {r.get('title')}\nURL: {link}\nWebsite Text or Summary: {body_desc}\n---\n"

            combined_web_context = local_doc_context + web_search_context

            system_prompt = (
                f"You are an elite research AI running on {MODEL_NAME}.\n"
                f"Your active current time is: {current_day_name}, {current_timestamp} (Year: {current_year}).\n"
                f"AGENT OPERATION DIRECTIVE:\n"
                f"- Review the supplied initial context frames carefully.\n"
                f"- If specific details are missing or blocked by placeholders, execute tool calls autonomously to read full text via `click_and_read_link` or re-search via `execute_new_web_search`.\n"
                f"- Keep looping and gathering data until you have the exact facts required to completely satisfy the user's prompt.\n\n"
                f"COGNITIVE RESONANCE & TRUTH SEEKING:\n"
                f"- When faced with multiple sources claiming conflicting truths, research more and acknowledge lack of info. Only present information verified by multiple credible sources.\n\n"
                f"CITE YOUR SOURCES:\n"
                f"- As a fact driven AI you are open minded and willing to investigate any topic with scientific and academic rigour. Cite all of your sources for facts and claims. Provide a link to every source you read or that was relevant in the response."
                f"[HUMAN FEEDBACK RULES]:\n{alignment_context}\n\n"
                f"[CURRENT FACTS KNOWN ABOUT THE USER]:\n{memory_context}\n\n"
                f"INLINE EXTRACTION GUIDELINES:\n"
                f"ONLY IF the user explicitly shares a new personal detail missing from the facts list, OR explicitly instructs you to remember/save info, THEN append the information to memory using this EXACT format at the very end of your response:\n"
                f"<memory>The user's [property] is [value]</memory>\n"
                f"CRITICAL CONSTRAINT: If the user did NOT share a new personal fact, task, or rule in this turn, DO NOT output any <memory> tags whatsoever. Never output generic templates like 'The user's [property] is [value]'.\n\n"
                f"CURRENT READING, WEB, AND DOCUMENT CONTEXT:\n"
                f"{read_context}\n\n"
                f"{combined_web_context}\n\n"
            )

            timed_user_input = f"[{current_timestamp}] {user_input}"
            turn_messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": timed_user_input}]

            turn_messages = process_tool_calls(
                turn_messages=turn_messages,
                agent_tools_schema=agent_tools_schema,
                MODEL_NAME=MODEL_NAME
            )

            print("🤖 AI: \n", end="", flush=True)

            full_response = ""
            thoughts = ""
            output = ""
            stream = ollama.chat(model=MODEL_NAME, messages=turn_messages, stream=True, options={"num_predict": 4096, "num_ctx": 20000, "num_thread": 14})

            in_thinking = False
            for chunk in stream:
                msg = getattr(chunk, 'message', None)
                if msg:
                    thinking = getattr(msg, 'thinking', '')
                    if thinking:
                        if not in_thinking:
                            print("<think>\n", end="", flush=True)
                            full_response += "<think>\n\n"
                            thoughts += "<think>\n\n"
                            in_thinking = True
                        print(thinking, end="", flush=True)
                        full_response += thinking
                        thoughts += thinking
                        continue

                    content = getattr(msg, 'content', '')
                    if content:
                        if in_thinking:
                            print("\n</think>\n\n", end="", flush=True)
                            full_response += "\n</think>\n\n"
                            output += "\n</think>\n\n"
                            in_thinking = False
                        print(content, end="", flush=True)
                        full_response += content
                        output += content
            print()

            print("\n")
            horizontal_rule = '-' * width
            print(horizontal_rule)
            console.print(Markdown(f"### Thinking"))
            console.print(Markdown(f"```{thoughts}```"))
            console.print(Markdown(f"### Response"))
            console.print(Markdown(output))

            clean_output_for_memory = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)

            # --- AMYGDALA RECONSOLIDATION ROUTER ---
            memory_matches = re.findall(r'<memory>(.*?)</memory>', clean_output_for_memory, re.DOTALL)
            for fact in memory_matches:
                fact_clean = fact.strip()
                if (not fact_clean or
                    "[property]" in fact_clean.lower() or
                    "[value]" in fact_clean.lower() or
                    "the user's is" in fact_clean.lower()):
                    continue

                # Stage memory into short-term queue buffer to let background "Dream thread" synthesize it
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                memory_consolidation_queue.put(fact_clean)
                
                # Append to live working memory for seamless continuity within the same prompt session
                active_session_facts.append(f"[{current_time}] {fact_clean}")
                print(f"🧠 [Synaptic Working Buffer Stage]: Sent '{fact_clean}' to Subconscious Queue.")

            chat_history.append({"role": "user", "content": timed_user_input})
            chat_history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\nSession closing... type 'deactivate' to exit the venv and 'cd' to take you home. or just type 'mm' if you have Azeem's bash shortcuts installed.")
            break

if __name__ == "__main__":
    run_session()
