# DREAMER: Introducing Cognitive AI Architecture.

An advanced, offline-first research AI system built with a biologically-inspired cognitive architecture. It operates entirely locally using Ollama for inference, Sentence Transformers for real-time text embeddings, and Qdrant as a multi-tier neural memory system.

***The core architecture moves beyond traditional RAG (Retrieval-Augmented Generation) by utilizing an asynchronous background loop that mimics human memory consolidation, semantic pruning, and dream cycles. Salience and consolidation during dreaming reduce mental clutter...***

**The implementation of *Cognitive AI Architecture* also serves as a useful demonstration of how Cognitive Resonance thinking systems lead to a reduction of doublethink, dissonance, and confusion, encourage critical thinking patterns, and lead to truth seeking behavior, regardless of whether the intelligence is organic or artificial.**

[For future discussion, consider Azeem's forthcoming book: The Theory of Cognitive Resonance, which was originally written pre AI, proven and demonstrated on Humans first, and later proven on AI as well.]

## Cognitive Architecture Features

*   **Three-Tier Memory Hierarchy:** Separates immediate context windows (Working Memory) from fast state tracking (Short-Term Cache) and optimized, long-term conceptual stores (Long Term Memory much like the Cerebral Cortex).
*   **Asynchronous "Dream" Consolidation Loop:** A continuous background thread processes raw, real-time insights, strips out formatting clutter, flattens overlapping data boundaries, and resolves logical contradictions.
*   **Refined Tool Handling & Researching:** Generates search engine queries using an independent keyword extractor and executes structural multi-turn tool loops over live data streams.
*   **Cognitive Resonance & Truth Seeking:** When faced with multiple sources claiming conflicting truths, it will either research more or acknowledge lack of info, and is instructed to only present information verified by multiple credible sources.
*   **Dynamic Variable Metadata Weighting:** Stores raw facts along with salience indicators, timestamps, and access trackers to establish cognitive importance patterns.
*   **Built-in Local Document Sharding:** Slices extensive textual materials into deep sliding overlapping window arrays using CPU-bound embedding matrices.

---

## System Prerequisites

Ensure the following system background engines are active before firing up the neural runtime script:

1.  **Ollama CLI**
   + Make sure `ollama ls` shows your model installed. Ollama by default runs on Port 11434. This system expects to find your ollama models on the default port.
   + To download a model, such as Qwen3:14B, use `ollama pull modelname`, for example: `ollama pull qwen3:14b`

2.  **Qdrant Vector Database** Make sure you have Qdrant's Docker Container Service running on Port 6333:

    ```bash
    docker run -p 6333:6333 qdrant/qdrant:latest 
    ```

---

## Installation & Local Environment Setup

1. **Open a New Terminal Session, and Clone the Repository Workspace to your home directory:**
   ```bash
   cd
   git clone https://github.com/azeemh/azeem_ai.git
   cd azeem_ai
   ```

2. **Initialize and Activate Python Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Interactive Console Command Guide

Run the application:
```bash
python3 azeem_ai_dreamer.py
```

If you'd like to run the original non-dreaming version (look at you, you aspiring scientist, ready to load the control version and everything! i love it...) use `azeem_ai.py`  

The workspace environment features multiline interface handling. Type complex prompt blocks freely and hit `Left ALT + Enter` to submit them.

| Terminal Operator | Architectural Trigger Function |
| :--- | :--- |
| `ingest path/to/book.md` | Reads documents into Qdrant shards with 500-line overlapping windows. |
| `read http://example.com` | instructs AI to read a website and adds the website's text to the current context. |
| `--doc` | Appends a contextual pipeline suffix. |
| `--web` | Enables search engine to fetch external real-time data. |
| `rlhf ` | Give AI reinforcement learning feedback... |
| `exit` / `quit` | Closes the session. |

---

## Manage Qdrant Vector Memory: 

Manage your Qdrant Memories Database by going to http://localhost:6333/ in your browser while the Qdrant docker container is running.

+ Quick Link to Memories Dashboard For Facts: http://localhost:6333/dashboard#/collections/fast_local_memories

---

### Azeem's Bash Shortcuts (optional shortcut commands for your CLI)

If you'd like to run **DREAMER** with the command `ai` 
1. open your .bashrc with vi: `vi ~/.bashrc` and scroll to the end with your keyboard arrows
2. press I to enter insert mode, then paste these aliases. (the `ai` alias shortcut assumes you installed to a folder in your user's home directory called `azeem_ai`)
```
#### Azeem's Bash Shortcuts ####

#azeem_ai shortcuts
alias ai="cd && cd azeem_ai && source ~/azeem_ai/venv/bin/activate && python3 azeem_ai_dreamer.py"

#parsertongue python shortcuts
alias activate="source venv/bin/activate"
alias mm="deactivate && cd"


```
3. Make sure you've pasted them correctly and quit with saving by pressing ESC, then press Z twice, then press ENTER. (to quit vi editor without saving press ESC, then type `:qa` or `:qa!`)
4. Afterwards you should be back at the command line. Type `source ~/.bashrc` to enable the use of your new bash shortcut in the current terminal session, or otherwise restart your terminal/start a new terminal session. 

Upon successfully installing the bash shortcut, you should be able to type **`ai`** into your terminal and run DREAMER.