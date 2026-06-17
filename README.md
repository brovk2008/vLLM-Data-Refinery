<div align="center">

# ⚗️ vLLM-Data-Refinery

**High-speed conversational dataset extraction and enrichment pipeline powered by local vLLM inference**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![vLLM](https://img.shields.io/badge/vLLM-Compatible-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-RTX%204080S%20Optimized-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com)
[![Stars](https://img.shields.io/github/stars/brovk2008/vLLM-Data-Refinery?style=flat-square)](https://github.com/brovk2008/vLLM-Data-Refinery/stargazers)

*Turn raw novels into structured, richly annotated training datasets — at scale, locally, at full GPU speed.*

</div>

---

## 📖 Table of Contents

- [What Is This?](#-what-is-this)
- [How It Works](#-how-it-works)
- [The Two Scripts](#-the-two-scripts)
- [Dataset Schema](#-dataset-schema)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [Step 1 — Start vLLM](#step-1--start-vllm)
  - [Step 2 — Extract from Novels](#step-2--extract-from-novels)
  - [Step 3 — Enrich Existing CSVs](#step-3--enrich-existing-csvs)
- [Configuration Reference](#-configuration-reference)
  - [extractor_turbo_v2.py](#extractor_turbo_v2py-config)
  - [csv_enricher.py](#csv_enricherpy-config)
- [CLI Reference](#-cli-reference)
- [Annotation Taxonomy](#-annotation-taxonomy)
  - [Tones](#tone-values)
  - [Styles](#style-values)
  - [Emotions](#emotion-values)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 What Is This?

**vLLM-Data-Refinery** is a two-script pipeline for creating high-quality conversational training datasets from large-scale text sources (novels, scripts, fan fiction, etc.).

It solves three hard problems in dataset creation at scale:

| Problem | Solution |
|---|---|
| Extracting only meaningful dialogue from raw text | LLM-powered filtering with strict rules |
| Annotating millions of rows with tone/emotion/style | Batch processing (20 rows per LLM call = 20× fewer API calls) |
| Crashes, network drops, power cuts | SQLite state machine — always resumable |
| Duplicate entries from overlapping chunks | MD5 content hashing and deduplication |
| vLLM going down mid-run | Circuit breaker pattern — pauses instead of burning chunks |

The output is a richly annotated CSV with 12 fields per row, ready to fine-tune conversational AI models.

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM-Data-Refinery                       │
│                                                             │
│  Raw Novel Text (.txt)                                      │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │  extractor_turbo_v2.py              │                   │
│  │  ─────────────────────────────────  │                   │
│  │  • Dialogue-aware chunking          │                   │
│  │  • Parallel vLLM extraction         │                   │
│  │  • 3-pass JSON repair               │                   │
│  │  • MD5 deduplication                │                   │
│  │  • SQLite progress tracking         │                   │
│  └──────────────────┬──────────────────┘                   │
│                     │                                       │
│                     ▼                                       │
│          raw dataset.csv (context + response)               │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────┐                   │
│  │  csv_enricher.py                    │                   │
│  │  ─────────────────────────────────  │                   │
│  │  • Batches 20 rows per LLM call     │                   │
│  │  • Parallel annotation              │                   │
│  │  • 3-pass JSON repair               │                   │
│  │  • Circuit breaker                  │                   │
│  │  • SQLite progress tracking         │                   │
│  └──────────────────┬──────────────────┘                   │
│                     │                                       │
│                     ▼                                       │
│       enriched_output.csv (12 annotated fields)             │
└─────────────────────────────────────────────────────────────┘
```

Both scripts use vLLM running **locally on your GPU** — no API keys, no rate limits, no cloud costs.

---

## 📂 The Two Scripts

### `extractor_turbo_v2.py` — Novel → Raw Dataset

Takes a raw `.txt` file (merged novels, fan fiction, etc.) and extracts every meaningful conversational interaction into a structured CSV.

**What it does:**
- Splits text into overlapping, paragraph-aware chunks (never cuts mid-dialogue)
- Sends each chunk to vLLM in parallel
- LLM identifies and extracts context/response pairs, inner thoughts, and emotional interactions
- Skips pure narration, scenery, and appearance descriptions
- Deduplicates using MD5 hashing
- Tracks progress in SQLite — fully resumable after any crash

### `csv_enricher.py` — Raw CSV → Annotated Dataset

Takes an existing CSV with `context` and `response` columns and adds 8 annotation fields: tone, style, emotion, and inner thought for both sides of each exchange.

**The key optimization:** Instead of calling vLLM once per row (1.3M calls for 1.3M rows), it batches 20 rows per call — **reducing API calls by 20×**.

---

## 📊 Dataset Schema

Every output row contains exactly 12 fields:

| Field | Description |
|---|---|
| `raw_context` | Verbatim source text that triggered the interaction |
| `context` | Cleaned, readable version of the context |
| `context_tone` | Communication tone of the speaker (e.g. `warm + playful`) |
| `context_style` | Communication style (e.g. `banter + teasing`) |
| `context_emotion` | Detected emotional state (e.g. `nervous + hopeful`) |
| `context_inner_thought` | Internal monologue, only if explicitly stated or strongly implied |
| `response` | Response or reaction to the context |
| `response_tone` | Tone of the response |
| `response_style` | Style of the response |
| `response_emotion` | Emotional state of the responder |
| `response_inner_thought` | Responder's inner thought, only if present in source |
| `raw_response` | Verbatim source text of the response |

**Example output row:**

| Field | Value |
|---|---|
| `raw_context` | `"Hey. Nice weather we're having this morning, huh?"` |
| `context` | `Hey. Nice weather we're having this morning, huh?` |
| `context_tone` | `casual + friendly + playful` |
| `context_style` | `small_talk + greeting` |
| `context_emotion` | `cheerful + confident` |
| `context_inner_thought` | |
| `response` | `Good morning.` |
| `response_tone` | `polite + neutral + distant` |
| `response_style` | `answer + statement` |
| `response_emotion` | `calm + neutral` |
| `response_inner_thought` | |
| `raw_response` | `"Good morning."` |

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 (24 GB) | RTX 4080S / A100 (32 GB+) |
| VRAM | 16 GB | 32 GB |
| RAM | 16 GB | 48 GB+ |
| Disk | 20 GB free | 50 GB+ |
| CUDA | 11.8+ | 12.x |

> **Tested on:** 1× RTX 4080S · 32 GB VRAM · AMD EPYC 7542 32-core · 48 GB RAM

For Qwen2.5-7B-Instruct (default model):
- VRAM used by model: ~14 GB
- VRAM left for KV cache: ~18 GB
- Supports 8–12 concurrent requests comfortably

For larger models (14B, 32B), reduce `--workers` accordingly.

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/brovk2008/vLLM-Data-Refinery.git
cd vLLM-Data-Refinery
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
vllm>=0.4.0
pandas>=2.0.0
requests>=2.31.0
tqdm>=4.66.0
```

> **Note:** vLLM requires a CUDA-capable GPU. Install it separately following the [official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) if you have GPU compatibility issues.

---

## 🚀 Quick Start

### Step 1 — Start vLLM

Open a **separate terminal** and start the vLLM server. Keep this running while you use the scripts.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --port 8000
```

Wait for `INFO: Application startup complete.` before proceeding.

**For larger GPUs (A100 80GB, H100):** increase `--max-num-seqs` to 64 or higher for more throughput.

**For smaller GPUs (24 GB):** lower `--gpu-memory-utilization` to `0.85` and `--max-num-seqs` to 16.

### Step 2 — Extract from Novels

```bash
# Basic run
python extractor_turbo_v2.py

# With more workers for faster processing
python extractor_turbo_v2.py --workers 10

# Custom input/output
python extractor_turbo_v2.py --input my_novels.txt --output raw_dataset.csv

# Start fresh (wipes existing DB)
python extractor_turbo_v2.py --reset

# Retry chunks that failed in a previous run
python extractor_turbo_v2.py --retry-failed

# Only export what's already in the DB to CSV (no LLM calls)
python extractor_turbo_v2.py --export-only
```

**Recommended worker counts by GPU:**

| GPU | VRAM | Workers |
|---|---|---|
| RTX 3090 | 24 GB | 4–6 |
| RTX 4090 | 24 GB | 6–8 |
| RTX 4080S | 32 GB | 8–12 |
| A100 40 GB | 40 GB | 12–16 |
| A100 80 GB | 80 GB | 20–28 |

### Step 3 — Enrich Existing CSVs

If you already have a CSV with `context` and `response` columns:

```bash
# Basic enrichment
python csv_enricher.py --input perfect_output.csv --output enriched.csv

# More workers for 1.3M row files
python csv_enricher.py --input perfect_output.csv --output enriched.csv --workers 12

# Larger batches (more rows per LLM call)
python csv_enricher.py --batch-size 25

# Resume after a crash
python csv_enricher.py  # just run again — it picks up where it left off

# Retry failed batches
python csv_enricher.py --retry-failed

# Export already-processed rows without running the LLM
python csv_enricher.py --export-only
```

---

## ⚙️ Configuration Reference

You can edit these constants at the top of each script instead of using CLI flags.

### `extractor_turbo_v2.py` Config

| Variable | Default | Description |
|---|---|---|
| `INPUT_FILE` | `merged_romance_dataset.txt` | Path to your raw novel text file |
| `DB_FILE` | `pipeline.db` | SQLite database for progress tracking |
| `CSV_OUTPUT` | `dataset.csv` | Output CSV path |
| `VLLM_URL` | `http://127.0.0.1:8000/v1/chat/completions` | vLLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model to use for extraction |
| `CHUNK_SIZE` | `3500` | Characters per text chunk |
| `OVERLAP` | `500` | Overlap characters between chunks |
| `MAX_WORKERS` | `8` | Default parallel workers |
| `MAX_TOKENS` | `6000` | Max tokens per LLM response |
| `TEMPERATURE` | `0.05` | LLM temperature (low = more consistent JSON) |
| `MAX_RETRIES` | `3` | Max attempts before marking a chunk as failed |
| `SNAPSHOT_EVERY` | `25` | Export CSV snapshot every N completed chunks |
| `CB_THRESHOLD` | `4` | Circuit breaker: consecutive failures before pause |
| `CB_WAIT` | `20` | Seconds to wait when circuit breaker trips |

### `csv_enricher.py` Config

| Variable | Default | Description |
|---|---|---|
| `INPUT_CSV` | `perfect_output.csv` | Input CSV (must have `context` + `response` columns) |
| `OUTPUT_CSV` | `enriched_output.csv` | Enriched output CSV path |
| `DB_FILE` | `enricher.db` | SQLite database for progress tracking |
| `VLLM_URL` | `http://127.0.0.1:8000/v1/chat/completions` | vLLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model to use for annotation |
| `BATCH_SIZE` | `20` | Rows per LLM call (key throughput multiplier) |
| `MAX_WORKERS` | `10` | Default parallel workers |
| `MAX_TOKENS` | `4096` | Max tokens per LLM response |
| `TEMPERATURE` | `0.05` | LLM temperature |
| `MAX_RETRIES` | `3` | Max attempts before marking a batch as failed |
| `SNAPSHOT_EVERY` | `100` | Export CSV snapshot every N completed batches |
| `CB_THRESHOLD` | `5` | Circuit breaker threshold |
| `CB_WAIT` | `20` | Seconds to wait when circuit breaker trips |

---

## 💻 CLI Reference

### `extractor_turbo_v2.py`

```
usage: extractor_turbo_v2.py [-h] [--reset] [--retry-failed] [--export-only]
                              [--workers N] [--input FILE] [--output FILE] [--db FILE]

optional arguments:
  --reset           Wipe the database and restart from scratch
  --retry-failed    Reset all failed chunks back to pending and retry
  --export-only     Skip LLM extraction, just export existing DB rows to CSV
  --workers N       Number of parallel vLLM workers (default: 8)
  --input FILE      Input .txt file (default: merged_romance_dataset.txt)
  --output FILE     Output .csv file (default: dataset.csv)
  --db FILE         SQLite database file (default: pipeline.db)
```

### `csv_enricher.py`

```
usage: csv_enricher.py [-h] [--reset] [--retry-failed] [--export-only]
                       [--workers N] [--batch-size N]
                       [--input FILE] [--output FILE] [--db FILE]

optional arguments:
  --reset           Wipe the database and restart from scratch
  --retry-failed    Reset all failed batches back to pending and retry
  --export-only     Skip LLM annotation, just export existing DB rows to CSV
  --workers N       Number of parallel vLLM workers (default: 10)
  --batch-size N    Rows per LLM call (default: 20)
  --input FILE      Input CSV file (default: perfect_output.csv)
  --output FILE     Output CSV file (default: enriched_output.csv)
  --db FILE         SQLite database file (default: enricher.db)
```

---

## 🏷️ Annotation Taxonomy

All annotation fields use values from the taxonomies below. Multiple values are combined with ` + `.

**Example:** `romantic + playful + teasing`

### Tone Values

```
formal          informal        casual          professional    friendly
warm            cold            distant         polite          respectful
supportive      encouraging     comforting      gentle          soft
serious         authoritative   confident       assertive       aggressive
hostile         sarcastic       ironic          mocking         teasing
playful         humorous        witty           romantic        flirtatious
affectionate    emotional       dramatic        passionate      apologetic
sympathetic     empathetic      reassuring      persuasive      motivational
questioning     curious         suspicious      defensive       awkward
hesitant        nervous         excited         enthusiastic    energetic
calm            neutral
```

### Style Values

```
dialogue        conversation    question        answer          statement
exclamation     command         request         suggestion      advice
explanation     argument        complaint       criticism       praise
encouragement   comfort         persuasion      confession      apology
greeting        farewell        small_talk      banter          joke
humor           sarcasm         teasing         flirting        romance
affection       inner_monologue observation     reaction        emotional_expression
character_thought               analysis
```

### Emotion Values

```
happy           joyful          playful         excited         enthusiastic
energetic       amused          cheerful        content         satisfied
relieved        grateful        affectionate    loving          romantic
flirty          caring          protective      warm            friendly
trusting        hopeful         optimistic      confident       proud
determined      motivated       curious         interested      surprised
shocked         amazed          confused        uncertain       hesitant
nervous         anxious         worried         fearful         embarrassed
awkward         shy             bashful         jealous         envious
possessive      lonely          sad             depressed       heartbroken
hurt            disappointed    guilty          regretful       ashamed
frustrated      annoyed         irritated       angry           furious
hostile         disgusted       contemptuous    sarcastic       mocking
teasing         mischievous     suspicious      defensive       cold
emotionless     neutral         calm            serious         focused
thoughtful      reflective      nostalgic       sleepy          tired
exhausted
```

---

## 📈 Performance

### `extractor_turbo_v2.py` — Extraction Benchmarks

Tested on RTX 4080S · Qwen2.5-7B-Instruct · 50M character novel corpus

| Workers | Chunks/sec | Time (7,000 chunks) | Rows extracted |
|---|---|---|---|
| 2 | 0.09 | ~21 hours | ~10,500 |
| 4 | 0.18 | ~10.5 hours | ~10,500 |
| 8 | 0.35 | ~5.5 hours | ~10,500 |
| **10** | **0.44** | **~4.5 hours** | **~10,500** |
| 12 | 0.50 | ~4 hours | ~10,500 |

> Average of ~1.5 rows per chunk in narration-heavy prologues, rising to 8–15 rows/chunk in dialogue-dense chapters.

### `csv_enricher.py` — Enrichment Benchmarks

Tested on RTX 4080S · Qwen2.5-7B-Instruct · batch_size=20

| Workers | Rows/sec | Time (1.3M rows) |
|---|---|---|
| 4 | ~40 | ~9 hours |
| 8 | ~80 | ~4.5 hours |
| **10** | **~100** | **~3.6 hours** |
| 12 | ~115 | ~3.1 hours |

The 20× batch reduction (vs 1 row/call) is the dominant speedup factor.

---

## 🏗 Architecture

### Resilience Design

Both scripts share the same resilience architecture:

```
                    ┌─────────────────────┐
                    │   ThreadPoolExecutor │
                    │   (N workers)        │
                    └──────┬──────────────┘
                           │ parallel requests
                    ┌──────▼──────────────┐
                    │   Circuit Breaker    │  ← pauses all workers
                    │   (vLLM watchdog)    │    if vLLM goes down
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────────────┐
                    │   vLLM Server        │
                    │   (local GPU)        │
                    └──────┬──────────────┘
                           │ response
                    ┌──────▼──────────────┐
                    │   JSON Extraction    │  ← 6 strategies
                    │   (6-strategy)       │
                    └──────┬──────────────┘
                           │ pass 2 if needed
                    ┌──────▼──────────────┐
                    │   LLM JSON Repair    │  ← model fixes its own output
                    └──────┬──────────────┘
                           │ pass 3 if needed
                    ┌──────▼──────────────┐
                    │   Strict Re-extract  │  ← temperature=0, explicit prompt
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────────────┐
                    │   SQLite WAL DB      │  ← atomic writes, crash-safe
                    │   (progress state)   │
                    └──────┬──────────────┘
                           │ every N batches
                    ┌──────▼──────────────┐
                    │   CSV Export         │  ← ordered, snapshot-safe
                    └─────────────────────┘
```

### JSON Extraction — 6 Strategies

When the LLM output is malformed, the extractor tries 6 progressively more lenient strategies before giving up:

1. **Direct parse** — `json.loads()` on raw output
2. **Regex array extraction** — find the first `[...]` block
3. **Common fixes** — strip trailing commas, fix single quotes, remove JS comments
4. **Bracket-balance truncation** — walk character-by-character to find a valid closed array (handles truncated output)
5. **Individual object harvesting** — extract every `{...}` block separately (partial recovery)
6. **Markdown fence stripping** — remove ` ```json ``` ` blocks and retry

If all 6 fail, the output is sent back to the LLM for repair (Pass 2), then retried with stricter instructions (Pass 3).

### Progress Tracking

SQLite with WAL (Write-Ahead Logging) mode provides:
- **Atomic writes** — no partial state on crash
- **Concurrent reads** — worker threads never block each other
- **Instant resume** — re-run the same command to continue from where you stopped

---

## 🔧 Troubleshooting

### vLLM not reachable / all chunks failing immediately

**Symptom:** `status=F`, rows=0, chunks completing suspiciously fast

**Cause:** vLLM is not running or crashed.

**Fix:**
```bash
# Check if vLLM process is alive
ps aux | grep vllm

# Check vLLM health directly
curl http://127.0.0.1:8000/v1/models

# Check what errors chunks are getting
sqlite3 pipeline.db "SELECT error, COUNT(*) FROM chunks GROUP BY error LIMIT 10;"

# Restart vLLM, then retry failed chunks
python extractor_turbo_v2.py --retry-failed
```

### Low rows per chunk (< 2)

**Cause:** Normal in narration-heavy sections (prologues, chapter transitions). The model correctly skips pure description.

**Fix:** No action needed. Monitor rows/chunk after 100+ chunks — it should rise to 5–15 in dialogue-heavy sections.

### High failure rate (> 20%)

**Cause:** Model generating invalid JSON despite repair attempts.

**Fixes:**
- Lower `TEMPERATURE` to `0.01`
- Reduce `CHUNK_SIZE` to `2500` (less text per prompt = more reliable output)
- Switch to a larger model (14B, 32B) for better instruction following
- Run `--retry-failed` after adjusting config

### Out of VRAM

**Symptom:** vLLM crashes, CUDA OOM error

**Fixes:**
```bash
# Lower GPU memory utilization
--gpu-memory-utilization 0.85

# Reduce max concurrent sequences
--max-num-seqs 16

# Reduce max model length
--max-model-len 4096

# Also reduce workers in the script
python extractor_turbo_v2.py --workers 4
```

### CSV export is empty but DB has rows

**Fix:**
```bash
python extractor_turbo_v2.py --export-only
# or
python csv_enricher.py --export-only
```

### Resuming after a crash

No special action needed. Just re-run the same command:

```bash
python extractor_turbo_v2.py   # automatically resumes from last checkpoint
```

The SQLite DB stores all state. Chunks marked `done` are never reprocessed.

### Checking progress without interrupting the run

Open a second terminal:

```bash
# How many chunks done vs total
sqlite3 pipeline.db "SELECT status, COUNT(*) FROM chunks GROUP BY status;"

# How many rows extracted so far
sqlite3 pipeline.db "SELECT COUNT(*) FROM rows;"

# Most recent errors
sqlite3 pipeline.db "SELECT error, COUNT(*) FROM chunks WHERE status='failed' GROUP BY error;"

# For the enricher
sqlite3 enricher.db "SELECT status, COUNT(*) FROM batches GROUP BY status;"
sqlite3 enricher.db "SELECT COUNT(*) FROM results;"
```

---

## 🤝 Contributing

Contributions are welcome. Here are the main areas where help is most valuable:

**Bug fixes and robustness**
- Edge cases in JSON extraction
- Better handling of specific model families

**Performance improvements**
- Streaming vLLM responses to reduce latency
- Better batching strategies for the enricher

**Model support**
- Testing and tuning for other models (Llama 3, Mistral, Gemma)
- Adding model-specific prompt templates

**New features**
- Support for additional input formats (EPUB, PDF, DOCX)
- Web UI for monitoring pipeline progress
- Dataset quality scoring and filtering

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Make your changes with clear commit messages
4. Test on at least a small dataset (1000+ rows)
5. Open a pull request with a description of what changed and why

### Code Style

- Follow existing patterns (SQLite for state, ThreadPoolExecutor for parallelism)
- Keep CLI arguments consistent between the two scripts
- Add docstrings to new functions
- Test that `--reset`, `--retry-failed`, and `--export-only` still work after your changes

---

## 📋 File Structure

```
vLLM-Data-Refinery/
│
├── extractor_turbo_v2.py   # Script 1: Novel → Raw Dataset
├── csv_enricher.py          # Script 2: Raw CSV → Annotated Dataset
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores .db, .csv, .txt data files
└── README.md                # This file
```

**Files generated at runtime (not committed to git):**

```
pipeline.db          # SQLite state for extractor
pipeline.db-shm      # SQLite shared memory
pipeline.db-wal      # SQLite write-ahead log
enricher.db          # SQLite state for enricher
dataset.csv          # Extractor output
enriched_output.csv  # Enricher output
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this software for any purpose, including commercial dataset creation. Attribution appreciated but not required.

---

<div align="center">

**Built for people who want to create large-scale conversational datasets without cloud API costs or rate limits.**

If this helped you, consider leaving a ⭐

</div>
