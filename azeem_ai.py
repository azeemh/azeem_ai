
"""
================================================──────────────────────────────_
LOCAL AGENTIC RAG CLI SYSTEM: "AZEEM-AI" README
================================================──────────────────────────────_

1. SYSTEM ARCHITECTURE DESCRIPTION
----------------------------------
This script initializes a 100% local, zero-bloat, privacy-hardened AI Agent
loop designed to mimic the multi-turn search behaviors of DeepSeek-R1 and
Gemini Pro. It operates completely offline.

2. CORE REASONING ENGINE PIPELINE
----------------------------------
- Pre-Scrape Harvester: Intercepts search queries and concurrently triggers
  both standard web (.text()) and live media news (.news()) APIs in parallel
  threads, deduplicating incoming URLs seamlessly.
- Semantic Node Condenser: Simulates a w3m CLI browser layout. Instead of
  bloated HTML source streams, it flattens DOM nodes into dense markdown blocks,
  saving up to 90% of your CPU context token processing metrics.
- Autonomous Tool Execution: Enables a recursive while-loop allowing the LLM
  to execute deep link scrapes or execute refined search parameters iteratively
  up to 5 times inside its own thinking loop before delivering answers.
- Dynamic Temporal Anchor: Forces raw timestamp evaluations live on every single
  user input turn to cleanly eliminate midnight context desynchronization.

3. HARDWARE DEPENDENCY REQUIREMENTS
------------------------------------
- OS Platform: Linux (Ubuntu/Debian tested)
- Python Engine: Python 3.10+ - (Tested with 3.12) (Requires modern wheels for PyTorch hooks)
- System VRAM Constraints: Tailored for an RTX 5070 with ~9032MiB allocated
- Ollama runtime slots. Employs aggressive CPU offloading for metadata generation. The idea is
  the gpu handles your model and to not clog your gpu, your cpu handles embeddings etc...

4. REQUIRED SOFTWARE & PYTHON DEPENDENCIES
-------------------------------------------
Run the following package manager scripts inside your targeted active (venv):

   $ pip install qdrant-client ollama sentence-transformers requests ddgs html2text

5. HOST SERVICES SETUP (BACKGROUND INFRASTRUCTURE CONTAINER DEPLOYMENTS)
------------------------------------------------------------------------
Before executing this script, ensure both backend server layers are active:

A. QDRANT VECTOR INSTANCE (Docker Container Service running on Port 6333):
   $ docker run -p 6333:6333 qdrant/qdrant:latest

B. OLLAMA RUNTIME LAYERS (Serving your reasoning target on Port 11434):
   $ ollama pull qwen3-next:80b-a3b-thinking
      or
   $ ollama pull qwen3:14b
      or
   $ ollama pull whatever_model_you_want_to_run

   MAKE SURE ollama ls SHOWS THAT YOUR MODEL IS INSTALLED SO YOU CAN CHOOSE IT FROM THE MENU. 
   It's advisable to install a model that knows how to use tool calls: for example, qwen3:14b and devstral:24b know how to use tool calls while deepseek-r1:14b cannot. 

6. LIVE APPLICATION INTERFACE CONTROLS
---------------------------------------
- Standard Prompting: Normal chat interaction. If trigger keywords match
  (e.g., search, latest, news, vs), parallel web pipelines fire instantly.
- Explicit Scrape: Type 'read <url>' to force an immediate full-page dump.
- Inline RLHF Matrix: Type 'rlhf' right after an unfavorable model response
  to write permanent behavioral rule metrics into the secondary Qdrant layer.
- System Close: Type 'exit' or 'quit' to safely exit the shell process loop.

7. Manage Qdrant Vector Memories: http://localhost:6333/

- Memories Dashboard For Facts: http://localhost:6333/dashboard#/collections/fast_local_memories

==============================================================================_

"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import uuid
import json
import threading
from datetime import datetime
import requests
import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
#from duckduckgo_search import DDGS
import html2text
import shutil
from prompt_toolkit import prompt

#rich markdown
from rich.markdown import Markdown
from rich.console import Console
console = Console()

def get_user_choice(options):
    """
    Displays a list of options and returns the user's selected choice.

    Args:
        options: List of strings representing available choices

    Returns:
        str: The selected option
    """
    while True:
        print("\nAvailable options:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))

            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid number. Please try again.")

        except ValueError:
            print("Please enter a valid number.")

def list_ollama_models():
    """Fetches and returns a list of model names from Ollama."""
    try:
        response = ollama.list()
        #print("Raw response from Ollama:", response)

        # Extract the nested list of models from the response object
        models_list = response.models if hasattr(response, 'models') else response.get('models', [])

        if not models_list:
            print("No models found. Please ensure Ollama is running and models are loaded.")
            return []

        # Safely extract the model string attribute from each Model object
        return [m.model for m in models_list]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

# --- DYNAMIC MODEL CONFIG -- #
model_names = list_ollama_models()

if model_names:
    selected_model = get_user_choice(model_names)
    print(f"\nYou selected the model: {selected_model}")
    # Use the selected_model for your application logic
    MODEL_NAME = selected_model
else:
    print("No models available to select.")
    MODEL_NAME = "qwen3:14b"


# --- CONFIGURATION ---
#MODEL_NAME = "qwen3:14b"
MEM_COLLECTION = "fast_local_memories"
RLHF_COLLECTION = "local_rlhf_rules"
DOC_COLLECTION = "document_shards"

# Get terminal width
width = shutil.get_terminal_size().columns

print("🚀 Loading instant 384-dim embedding model on CPU (Offline Mode)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

qdrant = QdrantClient(url="http://localhost:6333")

# Initialize all collections at 384 dimensions
for col in [MEM_COLLECTION, RLHF_COLLECTION, DOC_COLLECTION]:
    if not qdrant.collection_exists(col):
        from qdrant_client.models import Distance, VectorParams
        qdrant.create_collection(
            collection_name=col, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def get_embedding(text):
    return embed_model.encode(text).tolist()

def extract_and_execute_tool(model_output):
    # Regex to find the JSON structure even if wrapped in thinking text
    tool_call_match = re.search(r'\{"name":\s*"execute_new_web_search".*?\}', model_output, re.DOTALL)

    if tool_call_match:
        try:
            tool_call_json = json.loads(tool_call_match.group(0))
            query = tool_call_json["arguments"]["query"]
            print(f"Intercepted Tool Call! Querying: {query}")

            # Execute your search tool here using `query`
            # return execute_new_web_search(query)

        except json.JSONDecodeError:
            print("Found tool call but failed to decode JSON.")

    return None

def save_fact_async(collection, text, timestamp_str):
    try:
        # 1. Calculate the embedding
        vector = get_embedding(text)
        full_text = f"[{timestamp_str}] {text}"

        # 2. Generate a stable, unique ID based strictly on the core text content.
        # This guarantees that updating the same text overwrites the old entry.
        deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

        # 3. Use upsert to modify the entry in-place instead of uploading a duplicate
        qdrant.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=deterministic_id,
                    vector=vector,
                    payload={
                        "memory": full_text,
                        "timestamp": timestamp_str
                    }
                )
            ]
        )
        print(f"\n💾 [Stored/Updated] Metric/Rule locked: '{full_text}'")
    except Exception as e:
        print(f"\n❌ Storage error: {str(e)}")


def ingest_large_document(file_path):
    """Slices huge files into overlapping 500-line windows, resolving ~ paths safely."""
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
        return "\n".join([line.strip() for line in converter.handle(response.text).splitlines() if line.strip()])#[:2500] ##adding the [:2500] cuts off the websites it reads at 2500 characters.
    except Exception as e: return f"CLI Text download dropped: {str(e)}"

def run_parallel_hybrid_search(query):
    """Robust, flat-mapped search client that ensures valid dictionary payload formatting handles."""
    try:
        with DDGS() as ddgs:
            # Leverage flat-mapped generators to pull down clear text and instant answers cleanly
            text_hits = list(ddgs.text(query, max_results=4))

        blended_payloads = []
        for item in text_hits:
            if isinstance(item, dict):
                blended_payloads.append(item)
            else:
                # Handle native object schema properties via fallback attributes
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

            # Process tool call results
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

            # Add tool response to message history
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
    #load memory and rlhf
    m_points, _ = qdrant.scroll(collection_name=MEM_COLLECTION, limit=100, with_payload=True)
    r_points, _ = qdrant.scroll(collection_name=RLHF_COLLECTION, limit=100, with_payload=True)
    global_facts = [p.payload["memory"] for p in m_points if p.payload]
    rlhf_rules = [p.payload["memory"] for r in r_points if r.payload]
    chat_history = []

    print("=" * 80)
    print(f"🧠 Loaded {len(global_facts)} fact(s).")
    print(f"🎯 Loaded {len(rlhf_rules)} rlhf rule(s).")
    print(f"⚡ Loaded {MODEL_NAME}")
    print("UI Toggles per turn:\n  -> Ingest new documents with 'ingest path/to/file.md'\n  -> Read a url with 'read http://www.exampleurl.com' \n  -> Add '--doc' to tell AI to read local document vector shards & incorporate the documents into it's answer.\n  -> Add '--web' to execute real-time internet lookups.")
    print("  -> Multiline inputs are now supported so you have to press Left ALT + Enter to submit your input.")
    print("  -> Use CTRL + C to end the session.")
    print("=" * 80)


    while True:
        try:
            # FIXED: Initializing has_shards globally at top of turn loop prevents UnboundLocalError crashes
            has_shards = False
            memory_context = "\n".join(f"- {f}" for f in global_facts) if global_facts else "None"
            alignment_context = "\n".join(f"- {r}" for r in rlhf_rules) if rlhf_rules else "None"
            #user_raw_input = input("\n👤 You: \n").strip()
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
            today = now.strftime("Today is %A, %B %d, %Y, %I:%M:%S %p")

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

                print("\n🚨 [RLHF Mode] What did the model do wrong? Enter your corrective feedback:")
                feedback = input("👉 Feedback: ").strip()
                if feedback:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    formatted_rule = f"User Preference: {feedback}"
                    rlhf_rules.append(f"[{current_time}] {formatted_rule}")
                    save_fact_async(RLHF_COLLECTION, formatted_rule, current_time)
                    print("✅ Rule updated! Try your last prompt again.")
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

            # --- MODIFIED SYSTEM PROMPT ---
            system_prompt = (
                f"You are an elite research AI running on {MODEL_NAME}.\n"
                f"Your active current time is: {current_day_name}, {current_timestamp} (Year: {current_year}).\n"
                f"AGENT OPERATION DIRECTIVE:\n"
                f"- Review the supplied initial context frames carefully.\n"
                f"- If specific details are missing or blocked by placeholders, execute tool calls autonomously to read full text via `click_and_read_link` or re-search via `execute_new_web_search`.\n"
                f"- Keep looping and gathering data until you have the exact facts required to completely satisfy the user's prompt.\n\n"
                f"COGNITIVE RESONANCE & TRUTH SEEKING:\n"
                f"- When faced with multiple sources claiming conflicting truths, research more and acknowledge lack of info. Only present information verified by multiple credible sources.\n"
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

            # original settings num_predict": 2048, "num_ctx": 12288, "num_thread": 14

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

                        # This now executes for EVERY thinking chunk, including the first one
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

            #Markdown console output
            print("\n")
            # Create a horizontal rule
            horizontal_rule = '-' * width
            # Print it
            print(horizontal_rule)
            #print('Thinking:\n')
            console.print(Markdown(f"### Thinking"))
            console.print(Markdown(f"```{thoughts}```"))
            #print("\nResponse:\n")
            console.print(Markdown(f"### Response"))
            console.print(Markdown(output))


            # Remove everything inside <think>...</think> tags so internal reasoning is ignored
            clean_output_for_memory = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)

            # --- PYTHON VALIDATION GUARD ---
            # Look for memory tags inside the output
            memory_matches = re.findall(r'<memory>(.*?)</memory>', clean_output_for_memory, re.DOTALL)
            for fact in memory_matches:
                fact_clean = fact.strip()
                # Reject placeholders, empty entries, or template text
                if (not fact_clean or
                    "[property]" in fact_clean.lower() or
                    "[value]" in fact_clean.lower() or
                    "the user's is" in fact_clean.lower()):
                    continue # Skip saving this invalid memory

                # If valid, save it to Qdrant
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                global_facts.append(f"[{current_time}] {fact_clean}")
                save_fact_async(MEM_COLLECTION, fact_clean, current_time)

            chat_history.append({"role": "user", "content": timed_user_input})
            chat_history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\nSession closing... type 'deactivate' to exit the venv and 'cd' to take you home. or just type 'mm' if you have Azeem's bash shortcuts installed.")
            break

if __name__ == "__main__":
    run_session()
