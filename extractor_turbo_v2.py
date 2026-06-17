#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║      NOVEL DATASET EXTRACTOR — TURBO EDITION v2             ║
║      Optimized · RTX 4080S · 32GB VRAM · 32-core EPYC      ║
╠══════════════════════════════════════════════════════════════╣
║  BEFORE RUNNING — start vLLM in a separate terminal:        ║
║                                                              ║
║  python -m vllm.entrypoints.openai.api_server \             ║
║    --model Qwen/Qwen2.5-7B-Instruct \                       ║
║    --max-model-len 8192 \                                    ║
║    --gpu-memory-utilization 0.90 \                           ║
║    --max-num-seqs 32 \                                       ║
║    --port 8000                                               ║
║                                                              ║
║  Then run this script:                                       ║
║    python extractor_turbo_v2.py                              ║
║    python extractor_turbo_v2.py --workers 10                 ║
║    python extractor_turbo_v2.py --reset                      ║
║    python extractor_turbo_v2.py --export-only                ║
║    python extractor_turbo_v2.py --retry-failed               ║
╚══════════════════════════════════════════════════════════════╝
"""
__version__ = "1.0.0"
import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════
#  CONFIG  tuned for RTX 4080S / 32 GB VRAM / Qwen2.5-7B
# ════════════════════════════════════════════════════════════════

INPUT_FILE     = "merged_romance_dataset.txt"
DB_FILE        = "pipeline.db"
CSV_OUTPUT     = "dataset.csv"

VLLM_URL       = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_HEALTH    = "http://127.0.0.1:8000/v1/models"
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5-7B uses ~14 GB VRAM, leaving ~18 GB for KV cache = high concurrency
CHUNK_SIZE     = 3500     # chars per chunk
OVERLAP        = 500      # overlap so cross-paragraph exchanges arent lost
MAX_WORKERS    = 8        # parallel requests; tune 6-12 for your GPU
MAX_TOKENS     = 6000
TEMPERATURE    = 0.05     # near-deterministic = better JSON reliability
MAX_RETRIES    = 3
SNAPSHOT_EVERY = 25       # write CSV snapshot every N completed chunks

# Circuit breaker: pause instead of burning chunks when vLLM goes down
CB_THRESHOLD   = 4        # consecutive connection failures before pausing
CB_WAIT        = 20       # seconds to wait before re-checking vLLM


# ════════════════════════════════════════════════════════════════
#  SCHEMA
# ════════════════════════════════════════════════════════════════

COLUMNS = [
    "raw_context",
    "context",
    "context_tone",
    "context_style",
    "context_emotion",
    "context_inner_thought",
    "response",
    "response_tone",
    "response_style",
    "response_emotion",
    "response_inner_thought",
    "raw_response",
]


# ════════════════════════════════════════════════════════════════
#  PROMPTS
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert dataset creator for dialogue, emotion, and personality extraction from novels.

Extract high-quality conversational training examples from the provided text.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown. No explanations. No code blocks.
Must start with [ and end with ]. Must pass json.loads().

[{
  "raw_context":            "verbatim source text that triggered the interaction",
  "context":                "cleaned, readable version of context",
  "context_tone":           "tone1 + tone2",
  "context_style":          "style1 + style2",
  "context_emotion":        "emotion1 + emotion2",
  "context_inner_thought":  "only if explicitly stated or strongly implied, else empty string",
  "response":               "response or reaction to the context",
  "response_tone":          "tone1 + tone2",
  "response_style":         "style1 + style2",
  "response_emotion":       "emotion1 + emotion2",
  "response_inner_thought": "only if explicitly stated or strongly implied, else empty string",
  "raw_response":           "verbatim source text of the response"
}]

TONE VALUES (use + to combine):
formal | informal | casual | professional | friendly | warm | cold | distant |
polite | respectful | supportive | encouraging | comforting | gentle | soft |
serious | authoritative | confident | assertive | aggressive | hostile |
sarcastic | ironic | mocking | teasing | playful | humorous | witty |
romantic | flirtatious | affectionate | emotional | dramatic | passionate |
apologetic | sympathetic | empathetic | reassuring | persuasive | motivational |
questioning | curious | suspicious | defensive | awkward | hesitant | nervous |
excited | enthusiastic | energetic | calm | neutral

STYLE VALUES (use + to combine):
dialogue | conversation | question | answer | statement | exclamation |
command | request | suggestion | advice | explanation | argument | complaint |
criticism | praise | encouragement | comfort | persuasion | confession |
apology | greeting | farewell | small_talk | banter | joke | humor |
sarcasm | teasing | flirting | romance | affection | inner_monologue |
observation | reaction | emotional_expression | character_thought | analysis

EMOTION VALUES (use + to combine):
happy | joyful | playful | excited | enthusiastic | energetic | amused |
cheerful | content | satisfied | relieved | grateful | affectionate | loving |
romantic | flirty | caring | protective | warm | friendly | trusting |
hopeful | optimistic | confident | proud | determined | motivated | curious |
interested | surprised | shocked | amazed | confused | uncertain | hesitant |
nervous | anxious | worried | fearful | embarrassed | awkward | shy | bashful |
jealous | envious | possessive | lonely | sad | depressed | heartbroken |
hurt | disappointed | guilty | regretful | ashamed | frustrated | annoyed |
irritated | angry | furious | hostile | disgusted | contemptuous | sarcastic |
mocking | teasing | mischievous | suspicious | defensive | cold | emotionless |
neutral | calm | serious | focused | thoughtful | reflective | nostalgic |
sleepy | tired | exhausted

EXTRACTION RULES:
EXTRACT: dialogue exchanges, spoken responses, inner thoughts, emotional reactions,
         character interactions, feelings stated or narrated
SKIP:    pure scenery, appearance descriptions only, worldbuilding without interaction,
         exposition with no emotion or character response

QUALITY RULES:
- raw_context and raw_response = VERBATIM from source. Never alter them.
- Never invent dialogue, thoughts, or emotions not present in the source.
- Combine consecutive dialogue from the same exchange into ONE entry.
- Prefer 8 excellent entries over 20 mediocre ones.
- Maximum 20 objects per response.
- If output is getting long: STOP and close the array. Never leave an object half-open.
- A valid short array beats an invalid long one.
- ALL 12 fields required in every object. Use empty string when unavailable. Never use null.\
"""

REPAIR_SYSTEM = """\
You are a JSON repair expert.
Fix the broken JSON below and return ONLY the corrected JSON array.
Rules: start with [, end with ], no markdown, no explanations, change nothing except formatting.\
"""


# ════════════════════════════════════════════════════════════════
#  ANSI COLORS
# ════════════════════════════════════════════════════════════════

G   = "\033[92m"
Y   = "\033[93m"
R   = "\033[91m"
B   = "\033[94m"
W   = "\033[97m"
RST = "\033[0m"

def ok(msg):   print(f"  {G}OK{RST}  {msg}")
def warn(msg): print(f"  {Y}!!{RST}  {msg}")
def err(msg):  print(f"  {R}XX{RST}  {msg}")
def info(msg): print(f"  {B}->{RST}  {msg}")


# ════════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER
#  Pauses all workers if vLLM drops mid-run instead of burning chunks
# ════════════════════════════════════════════════════════════════

class CircuitBreaker:
    def __init__(self):
        self._lock     = Lock()
        self._failures = 0
        self._open     = False

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._open     = False

    def record_conn_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= CB_THRESHOLD:
                self._open = True

    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def wait_if_open(self, pbar=None):
        if not self.is_open():
            return
        msg = f"\n  vLLM unreachable — pausing {CB_WAIT}s then retrying..."
        if pbar:
            pbar.write(Y + msg + RST)
        else:
            print(Y + msg + RST)
        time.sleep(CB_WAIT)
        if check_vllm():
            with self._lock:
                self._failures = 0
                self._open     = False
            msg2 = "  vLLM back online — resuming."
            if pbar:
                pbar.write(G + msg2 + RST)
            else:
                print(G + msg2 + RST)


BREAKER = CircuitBreaker()


# ════════════════════════════════════════════════════════════════
#  VLLM HEALTH
# ════════════════════════════════════════════════════════════════

def check_vllm() -> bool:
    try:
        r = requests.get(VLLM_HEALTH, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_vllm(timeout: int = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if check_vllm():
            return True
        time.sleep(5)
    return False


# ════════════════════════════════════════════════════════════════
#  DATABASE  SQLite WAL — atomic, crash-safe, resumable
# ════════════════════════════════════════════════════════════════

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id       INTEGER PRIMARY KEY,
            text     TEXT    NOT NULL,
            status   TEXT    DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            error    TEXT    DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS rows (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id   INTEGER NOT NULL,
            hash       TEXT    UNIQUE,
            data       TEXT    NOT NULL,
            exported   INTEGER DEFAULT 0,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
        CREATE INDEX IF NOT EXISTS idx_rows_exported ON rows(exported);
        CREATE INDEX IF NOT EXISTS idx_rows_chunk    ON rows(chunk_id);
    """)
    conn.commit()
    return conn


def reset_db(conn: sqlite3.Connection):
    conn.executescript("DELETE FROM rows; DELETE FROM chunks;")
    conn.commit()
    warn("Database wiped — starting fresh.")


def retry_failed_chunks(conn: sqlite3.Connection, lock: Lock):
    with lock:
        n = conn.execute(
            "UPDATE chunks SET status='pending', attempts=0, error='' WHERE status='failed'"
        ).rowcount
        conn.commit()
    info(f"Reset {n} failed chunks back to pending.")


def populate_chunks(conn: sqlite3.Connection, chunks: list) -> None:
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO chunks(id, text, status) VALUES (?, ?, 'pending')",
            [(i, c) for i, c in enumerate(chunks)],
        )
        conn.commit()
        ok(f"Inserted {len(chunks)} chunks.")
    else:
        done    = conn.execute("SELECT COUNT(*) FROM chunks WHERE status='done'").fetchone()[0]
        failed  = conn.execute("SELECT COUNT(*) FROM chunks WHERE status='failed'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM chunks WHERE status='pending'").fetchone()[0]
        info(f"Resuming: {done} done / {pending} pending / {failed} failed of {count} total.")


def get_pending(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT id, text FROM chunks WHERE status IN ('pending','failed') AND attempts < ?",
        (MAX_RETRIES,),
    ).fetchall()


def mark_chunk(conn: sqlite3.Connection, lock: Lock,
               cid: int, status: str, error: str = ""):
    with lock:
        conn.execute(
            "UPDATE chunks SET status=?, attempts=attempts+1, error=? WHERE id=?",
            (status, error[:300], cid),
        )
        conn.commit()


def insert_rows(conn: sqlite3.Connection, lock: Lock,
                chunk_id: int, rows: list) -> int:
    inserted = 0
    with lock:
        for row in rows:
            h = hashlib.md5(
                (row.get("raw_context", "") + "||" + row.get("raw_response", "")).encode()
            ).hexdigest()
            try:
                conn.execute(
                    "INSERT INTO rows(chunk_id, hash, data) VALUES (?, ?, ?)",
                    (chunk_id, h, json.dumps(row, ensure_ascii=False)),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # duplicate — skip silently
        conn.commit()
    return inserted


def export_to_csv(conn: sqlite3.Connection, lock: Lock, out_path: str) -> int:
    with lock:
        rows = conn.execute(
            "SELECT id, data FROM rows WHERE exported=0 ORDER BY id"
        ).fetchall()
    if not rows:
        return 0
    records, ids = [], []
    for rid, data in rows:
        try:
            obj = json.loads(data)
            records.append({col: str(obj.get(col) or "") for col in COLUMNS})
            ids.append(rid)
        except Exception:
            pass
    if not records:
        return 0
    df = pd.DataFrame(records, columns=COLUMNS)
    file_exists = Path(out_path).exists()
    df.to_csv(out_path, mode="a", header=not file_exists, index=False, encoding="utf-8")
    with lock:
        conn.execute(
            "UPDATE rows SET exported=1 WHERE id IN ({})".format(
                ",".join(["?"] * len(ids))
            ),
            ids,
        )
        conn.commit()
    return len(records)


# ════════════════════════════════════════════════════════════════
#  TEXT PROCESSING  dialogue-aware chunking
# ════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_chunks(text: str) -> list:
    """
    Split at paragraph boundaries — never mid-sentence or mid-dialogue.
    Carry trailing overlap so cross-paragraph exchanges are preserved.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks, current, cur_len = [], [], 0

    for para in paragraphs:
        plen = len(para)
        if cur_len + plen > CHUNK_SIZE and current:
            chunks.append("\n\n".join(current))
            # carry overlap: walk backwards, collect up to OVERLAP chars
            overlap_buf, overlap_len = [], 0
            for p in reversed(current):
                if overlap_len + len(p) + 2 <= OVERLAP:
                    overlap_buf.insert(0, p)
                    overlap_len += len(p) + 2
                else:
                    break
            current = overlap_buf
            cur_len = overlap_len
        current.append(para)
        cur_len += plen + 2

    if current:
        chunks.append("\n\n".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ════════════════════════════════════════════════════════════════
#  JSON EXTRACTION  6 strategies, most to least lenient
# ════════════════════════════════════════════════════════════════

def _parse_list(text: str) -> Optional[list]:
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else None
    except Exception:
        return None


def _parse_dict(s: str) -> Optional[dict]:
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def extract_json(text: str) -> Optional[list]:
    text = text.strip()

    # 1 Direct parse
    r = _parse_list(text)
    if r is not None:
        return r

    # 2 Grab first [...] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        r = _parse_list(m.group(0))
        if r is not None:
            return r

    # 3 Common JSON fixes: trailing commas, single quotes, JS comments
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    fixed = re.sub(r"//.*?\n", "\n", fixed)
    m = re.search(r"\[.*\]", fixed, re.DOTALL)
    if m:
        r = _parse_list(m.group(0))
        if r is not None:
            return r

    # 4 Bracket-balance truncation (handles LLM-truncated output)
    try:
        start = text.index("[")
        depth, last_close = 0, -1
        in_string, escape = False, False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    last_close = i
                    break
        if last_close > start:
            r = _parse_list(text[start:last_close + 1])
            if r is not None:
                return r
    except ValueError:
        pass

    # 5 Harvest individual objects (partial recovery)
    objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    if objects:
        valid = [d for s in objects for d in [_parse_dict(s)] if d is not None]
        if valid:
            return valid

    # 6 Strip markdown fences and retry
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    r = _parse_list(stripped)
    if r is not None:
        return r
    m = re.search(r"\[.*\]", stripped, re.DOTALL)
    if m:
        r = _parse_list(m.group(0))
        if r is not None:
            return r

    return None


# ════════════════════════════════════════════════════════════════
#  ROW VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_row(row: dict) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    fixed = {col: str(row.get(col) or "") for col in COLUMNS}
    # Reject completely empty rows
    if not fixed["raw_context"].strip() and not fixed["response"].strip():
        return None
    # Best-effort fallback for missing verbatim fields
    if fixed["context"].strip() and not fixed["raw_context"].strip():
        fixed["raw_context"] = fixed["context"]
    if fixed["response"].strip() and not fixed["raw_response"].strip():
        fixed["raw_response"] = fixed["response"]
    return fixed


# ════════════════════════════════════════════════════════════════
#  vLLM CLIENT  per-thread session, exponential back-off
# ════════════════════════════════════════════════════════════════

_tls = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_tls, "session"):
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json"})
        _tls.session = s
    return _tls.session


def call_llm(messages: list, temperature: float = TEMPERATURE,
             max_tokens: int = MAX_TOKENS) -> Optional[str]:
    session = get_session()
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    for attempt in range(3):
        try:
            resp = session.post(VLLM_URL, json=payload, timeout=300)
            resp.raise_for_status()
            BREAKER.record_success()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            BREAKER.record_conn_failure()
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            time.sleep(10 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code in (429, 503):
                time.sleep(15 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(5)
    return None


# ════════════════════════════════════════════════════════════════
#  CHUNK PROCESSING  3-pass extraction with repair
# ════════════════════════════════════════════════════════════════

def process_chunk(chunk_id: int, chunk_text: str,
                  conn: sqlite3.Connection, db_lock: Lock,
                  pbar=None) -> tuple:
    """Returns (rows_inserted: int, status: str)"""

    # Pause here if vLLM is unreachable
    BREAKER.wait_if_open(pbar)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"TEXT:\n\n{chunk_text}"},
    ]

    # Pass 1: Primary generation
    output = call_llm(messages)
    if not output:
        mark_chunk(conn, db_lock, chunk_id, "failed", "NO_LLM_RESPONSE")
        return 0, "failed"

    data = extract_json(output)

    # Pass 2: LLM-assisted JSON repair
    if data is None:
        repair_output = call_llm(
            [{"role": "system", "content": REPAIR_SYSTEM},
             {"role": "user",   "content": output}],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        if repair_output:
            data = extract_json(repair_output)

    # Pass 3: Strict re-extraction
    if data is None:
        retry_output = call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"TEXT:\n\n{chunk_text}\n\n"
                    "IMPORTANT: Return ONLY a valid JSON array. "
                    "Start with [. End with ]. "
                    "Even an empty array [] is acceptable. No other text."
                )},
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        if retry_output:
            data = extract_json(retry_output)

    if not data or not isinstance(data, list):
        mark_chunk(conn, db_lock, chunk_id, "failed", "INVALID_JSON_ALL_PASSES")
        return 0, "failed"

    # Validate rows
    valid = [v for row in data for v in [validate_row(row)] if v is not None]

    inserted = insert_rows(conn, db_lock, chunk_id, valid)
    mark_chunk(conn, db_lock, chunk_id, "done")
    return inserted, "done"


# ════════════════════════════════════════════════════════════════
#  STATS REPORT
# ════════════════════════════════════════════════════════════════

def print_stats(conn: sqlite3.Connection, elapsed: float):
    s = conn.execute("""
        SELECT
            SUM(CASE status WHEN 'done'    THEN 1 ELSE 0 END),
            SUM(CASE status WHEN 'failed'  THEN 1 ELSE 0 END),
            SUM(CASE status WHEN 'pending' THEN 1 ELSE 0 END),
            COUNT(*)
        FROM chunks
    """).fetchone()
    total_rows    = conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
    exported_rows = conn.execute("SELECT COUNT(*) FROM rows WHERE exported=1").fetchone()[0]
    dupes         = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT hash) FROM rows"
    ).fetchone()[0]
    done, failed, pending, total = [x or 0 for x in s]
    mins, secs = divmod(int(elapsed), 60)

    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  EXTRACTION REPORT")
    print(sep)
    print(f"  Time elapsed         {mins}m {secs}s")
    print(f"  Chunks done          {G}{done}{RST}")
    print(f"  Chunks failed        {R}{failed}{RST}")
    print(f"  Chunks pending       {Y}{pending}{RST}")
    print(f"  Chunks total         {total}")
    print(f"  Rows extracted       {W}{total_rows}{RST}")
    print(f"  Duplicates skipped   {dupes}")
    print(f"  Rows exported        {G}{exported_rows}{RST}")
    if done:
        print(f"  Avg rows / chunk     {total_rows / done:.1f}")
    if total:
        pct = done / total * 100
        color = G if pct >= 90 else Y if pct >= 70 else R
        print(f"  Success rate         {color}{pct:.1f}%{RST}")
    print(sep + "\n")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Novel to Conversational Dataset Extractor — Turbo v2"
    )
    p.add_argument("--reset",        action="store_true",
                   help="Wipe database and restart from scratch")
    p.add_argument("--retry-failed", action="store_true",
                   help="Reset all failed chunks to pending and retry")
    p.add_argument("--export-only",  action="store_true",
                   help="Skip extraction — just export rows to CSV")
    p.add_argument("--workers",      type=int, default=MAX_WORKERS,
                   help=f"Parallel workers (default {MAX_WORKERS}, sweet spot 8-12 for RTX 4080S)")
    p.add_argument("--input",        type=str, default=INPUT_FILE)
    p.add_argument("--output",       type=str, default=CSV_OUTPUT)
    p.add_argument("--db",           type=str, default=DB_FILE)
    p.add_argument("--model", type=str, default=MODEL_NAME,
               help=f"vLLM model name (default: {MODEL_NAME})")
    p.add_argument("--url", type=str, default=VLLM_URL,
               help="vLLM API URL (default: localhost:8000)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    sep  = "=" * 54
    MODEL_NAME = args.model
    VLLM_URL = args.url
    print(f"\n{sep}")
    print(f"  {W}NOVEL DATASET EXTRACTOR  TURBO EDITION v2{RST}")
    print(f"  Optimized for RTX 4080S  32 GB VRAM")
    print(sep + "\n")

    # Database setup
    conn    = init_db(args.db)
    db_lock = Lock()

    if args.reset:
        reset_db(conn)

    if args.retry_failed:
        retry_failed_chunks(conn, db_lock)

    # Export-only mode
    if args.export_only:
        info("Export-only mode.")
        n = export_to_csv(conn, db_lock, args.output)
        ok(f"Exported {n} rows to {args.output}")
        print_stats(conn, 0)
        conn.close()
        return

    # vLLM health check
    print(f"  Checking vLLM at {VLLM_URL}...")
    if not check_vllm():
        err("vLLM is NOT reachable!")
        print(f"""
  {Y}Start vLLM first in a SEPARATE terminal:{RST}

    python -m vllm.entrypoints.openai.api_server \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --max-model-len 8192 \\
      --gpu-memory-utilization 0.90 \\
      --max-num-seqs 32 \\
      --port 8000

  Waiting up to 2 minutes for vLLM to come up...
""")
        if not wait_for_vllm(120):
            err("vLLM still not reachable. Exiting.")
            sys.exit(1)
    ok("vLLM is online.\n")

    # Load and chunk text
    input_path = Path(args.input)
    if not input_path.exists():
        err(f"Input file not found: {args.input}")
        sys.exit(1)

    info(f"Loading: {args.input}")
    text   = input_path.read_text(encoding="utf-8", errors="ignore")
    text   = clean_text(text)
    chunks = create_chunks(text)
    info(f"Text: {len(text):,} chars  ->  {len(chunks):,} chunks")
    print()

    populate_chunks(conn, chunks)
    pending = get_pending(conn)

    if not pending:
        warn("No pending chunks. Use --reset or --retry-failed to reprocess.")
        n = export_to_csv(conn, db_lock, args.output)
        ok(f"Exported {n} rows to {args.output}")
        print_stats(conn, 0)
        conn.close()
        return

    info(f"Chunks to process: {len(pending)}")
    info(f"Workers: {args.workers}  (RTX 4080S sweet spot: 8-12)")
    info(f"Chunk size: {CHUNK_SIZE} chars")
    print()

    # Parallel extraction
    start_time   = time.time()
    total_rows   = 0
    snap_counter = 0
    fail_counter = 0

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    with tqdm(total=len(pending), desc="Chunks", unit="chunk",
              ncols=85, bar_format=bar_fmt) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_chunk, cid, ctxt, conn, db_lock, pbar): cid
                for cid, ctxt in pending
            }
            for future in as_completed(future_map):
                cid = future_map[future]
                try:
                    inserted, status = future.result()
                    total_rows   += inserted
                    snap_counter += 1
                    if status == "failed":
                        fail_counter += 1

                    pbar.set_postfix(
                        rows=total_rows,
                        fail=fail_counter,
                        s=status[0].upper(),
                        refresh=False,
                    )

                    # Periodic CSV snapshot
                    if snap_counter >= SNAPSHOT_EVERY:
                        n = export_to_csv(conn, db_lock, args.output)
                        snap_counter = 0
                        if n:
                            pbar.write(f"  Snapshot: {n} rows -> {args.output}")

                except Exception as e:
                    mark_chunk(conn, db_lock, cid, "failed", str(e)[:200])
                    fail_counter += 1

                pbar.update(1)

    # Final CSV export
    print()
    info("Final CSV export...")
    n = export_to_csv(conn, db_lock, args.output)
    ok(f"Exported {n} rows to {args.output}")

    print_stats(conn, time.time() - start_time)
    conn.close()


if __name__ == "__main__":
    main()