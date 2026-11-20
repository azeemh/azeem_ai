# DREAMER: Introducing Cognitive AI Architecture.

An advanced, offline-first research AI system built with a biologically-inspired cognitive architecture. It operates entirely locally using Ollama for inference, Sentence Transformers for real-time text embeddings, and Qdrant as a multi-tier neural memory system.

**The core architecture moves beyond traditional RAG (Retrieval-Augmented Generation) by utilizing an asynchronous background loop that mimics human memory consolidation, semantic pruning, and dream cycles.**

***Salience focuses attention, and consolidation during dreaming reduces mental clutter...***

The implementation of *Cognitive AI Architecture* also serves as a useful demonstration of how **Cognitive Resonance thinking systems lead to a reduction of doublethink, dissonance, and confusion, encourage critical thinking patterns, and lead to truth seeking behavior, regardless of whether the intelligence is organic or artificial.**

[For future discussion, consider Azeem's forthcoming book: The Theory of Cognitive Resonance, originally coining the term pre AI, proving and demonstrating the idea on Humans first, and now with this software library demonstrating it with AI as well.]

## Cognitive Architecture Features

*   **Three-Tier Memory Hierarchy:** Separates immediate context windows (Working Memory) from fast state tracking (Short-Term Cache) and optimized, long-term conceptual stores (Long Term Memory much like the Cerebral Cortex).
*   **Asynchronous "Dream" Consolidation Loop:** A background thread waits for 60 seconds of quiet, then processes raw insights — stripping formatting clutter, flattening overlapping data boundaries, and resolving logical contradictions. Active conversations are never interrupted.
*   **Refined Tool Handling & Researching:** Generates search engine queries using an independent keyword extractor and executes structural multi-turn tool loops over live data streams.
*   **Cognitive Resonance & Truth Seeking:** When faced with multiple sources claiming conflicting truths, it will either research more or acknowledge lack of info, and is instructed to only present information verified by multiple credible sources.
*    **Salience-Weighted Memory:** Every fact carries a salience score (1–10). Retrieval fetches the top ten cosine matches, reranks with 0.65 × similarity + 0.35 × salience, and injects the top three. Retrieved facts get +1 salience; untouched facts decay by 1 per week (floor 1); superseded facts are demoted to 2.
*   **Multi-Document Ingestion:** `ingest path/to/file` slices any document into sliding-window character shards using Ollama-served embeddings on CPU. Multiple documents coexist; appending --doc to the end of a query searches across all of them, and --docs lists what's in the corpus.
*   **Single-Model Reasoning:** One chat model instance handles chat, tool calls, keyword extraction, and consolidation. Load parameters are pinned so Ollama reuses the same loaded instance across call sites — no reloads between a tool call and its streamed answer. This optimization allows higher accuracy and one instance to stay in memory on your GPU, especially useful for folks optimizing VRAM use...
*   **CPU-Only Embeddings:** The embedding model (`qwen3-embedding:4b`, 2560-dim) is served by Ollama with `num_gpu: 0`, keeping VRAM fully reserved for the chat model.
---
### Quick Start Guide:

First Time Users: 
1. Install Ollama
2. Have a Model Running
3. Install Docker
4. Start Docker Container as instructed below in System Prerequisites (Section 2).
5. Complete the **Installation and Local Environment Setup** by cloning the repo, initializing the virtual environment, and installing the dependencies.
6. Run the app as specified in **Interactive Console Command Guide**

#### Returning Users:
1. Make sure the Qdrant Container is running 
```bash 
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant-storage:/qdrant/storage qdrant/qdrant
```
2. Run the script
```bash
python3 azeem_ai_dreamer.py
```
`python` may be the default if your computer never had to deal with the python2 and python3 fiasco.

#### If you installed my shortcuts:
1. Type `qdrant` and press ENTER to start qdrant if it already is not running.
2. Type `ai` and press ENTER to get started.

---
## System Prerequisites

Ensure the following system background engines are active before firing up the neural runtime script:

1.  **Native Ollama CLI already installed & running**
   + If you dont have ollama installed follow the instructions here: https://ollama.com/download 
   + Make sure `ollama ls` shows your model. Ollama by default runs on Port 11434. DREAMER expects to find your ollama models on the default port.
   + To download a model, such as Qwen3:14B, use `ollama pull modelname`, for example: `ollama pull qwen3:14b`
   + You MUST install an embeddings model. 
   ```bash
   ollama pull qwen3-embedding:4b
   ```

2.  **Qdrant Vector Database Docker Container already running** 
   + a. First make a local volume so your qdrant memories survive a restart (this way you can close qdrant and re-open a new session): 
   ```bash
   docker volume create qdrant-storage
   ``` 
   + b. Then Make sure you have Qdrant's Docker Container Service running on Port 6333:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 -v qdrant-storage:/qdrant/storage qdrant/qdrant
   ```
   + c. **Once you have Azeem's Bash Shortcuts installed you can just type:** 
   ```bash
   qdrant
   ```
   + ***Only use this command instead if you'd like to use ephemeral container memory that wipes after shutdown/restart:*** 

   `docker run -p 6333:6333 qdrant/qdrant:latest`

   + NOTE: advanced users will know that `docker run -d -p 6333:6333 -p 6334:6334 -v qdrant-storage:/qdrant/storage qdrant/qdrant` from STEP 2B will automatically create the qdrant-storage volume if it doesn't exist, however the command isn't very easy to parse at a glance for humans, users, and normal people to know what's going on, so I explicitly included the command to create a volume. Users can also create new volumes and modify the -v flag to correspond to the new volume, thus allowing them to run the qdrant system with different memory bank volumes. Thus a researcher can run the dreamer and control system on various models, and compare model performance across various memory bank collections, explore memory deletion implications, and more...

   + ⚠️ **Upgrading from the old 384-dim version:** if you previously ran an older version of DREAMER with `sentence_transformers`, your Qdrant collections use 384-dim vectors and the new script uses 2560-dim. On first launch, the script prints a `[Migration]` line and **deletes + recreates** each collection. Old memories, RLHF rules, and ingested documents are lost. Please Export via the Qdrant dashboard first if you want to keep them.
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

If you'd like to run the original non-dreaming version use `azeem_ai.py` -- (look at you, you aspiring scientist, ready to load the control version and everything! i love it...)

The workspace environment features multiline interface handling. Type complex prompt blocks freely and hit `Left ALT + Enter` to submit them.

| Terminal Operator | Architectural Trigger Function |
| :--- | :--- |
| `ingest path/to/book.md` | Reads any doc/file, saves as qdrant shards. |
| `ls docs`| Lists ingested documents
| `read http://www.example.com` | Reads any website text & adds it to context. |
| `--doc` | Enables AI to access ingested documents. |
| `--web` | Enables AI to search external real-time data. |
| `rlhf ` | Give AI reinforcement learning feedback... |
| `exit` / `quit` | Closes the session. |
---
## Manage Qdrant Vector Memory: 

Manage your Qdrant Memories Database by going to http://localhost:6333/ in your browser while the Qdrant docker container is running.

+ Quick Link to Memories Dashboard For Facts: http://localhost:6333/dashboard#/collections/fast_local_memories

---

### Azeem's Bash Shortcuts (optional shortcut commands for your CLI)

If you'd like to run **DREAMER** with the command `ai` 
1. (OG Linux, Posix, Mac OS users, anything with a bash terminal) open your .bashrc with vi: `vi ~/.bashrc` and scroll to the end with your keyboard arrows | ZSH users & Mac OS Catalina+ users open .zshrc `vi ~/.zshrc` as they use zsh terminal and likewise scroll to the end.
2. press I to enter insert mode, then paste these aliases. (the `ai` alias shortcut assumes you installed to a folder in your user's home directory called `azeem_ai` which you should have by default if you followed the installation instructions above.)
```
#### Azeem's Bash Shortcuts ####

#azeem_ai shortcuts
alias qdrant="docker run -d -p 6333:6333 -p 6334:6334 -v qdrant-storage:/qdrant/storage qdrant/qdrant"
alias ai="cd && cd azeem_ai && source ~/azeem_ai/venv/bin/activate && python3 azeem_ai_dreamer.py"

#parsertongue python shortcuts
alias activate="source venv/bin/activate"
alias mm="deactivate && cd"


```
3. Make sure you've pasted them correctly and quit with saving by pressing ESC, then press Z twice, then press ENTER. (to quit vi editor without saving press ESC, then type `:qa` or `:qa!`)
4. Afterwards you should be back at the command line. Type `source ~/.bashrc` (or `source ~/.zshrc`) to enable the use of your new shortcut in the current terminal session, or otherwise restart your terminal/start a new terminal session. 

Upon successfully installing the bash shortcuts, you should be able to type **`qdrant`** into your terminal and start the memory server. Then you should be able to type **`ai`** into your terminal and run DREAMER.
