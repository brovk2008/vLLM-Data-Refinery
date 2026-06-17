#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   CSV ENRICHMENT PIPELINE — TURBO                           ║
║   Adds tone · style · emotion · inner_thought               ║
║   to existing context + response CSV                        ║
║   Optimized · RTX 4080S · 32GB VRAM · 1.3M rows            ║
╠══════════════════════════════════════════════════════════════╣
║  HOW IT WORKS:                                              ║
║  Instead of 1 LLM call per row (1.3M calls),               ║
║  it sends 20 rows per call  =  ~65K calls total             ║
║  That is a 20x speedup before even counting parallelism.    ║
║                                                             ║
║  USAGE:                                                     ║
║    python csv_enricher.py                                   ║
║    python csv_enricher.py --workers 12                      ║
║    python csv_enricher.py --batch-size 25                   ║
║    python csv_enricher.py --reset                           ║
║    python csv_enricher.py --retry-failed                    ║
║    python csv_enricher.py --export-only                     ║
║    python csv_enricher.py --input myfile.csv --output out.csv ║
╚══════════════════════════════════════════════════════════════╝
"""
__version__ = "1.0.0"
import argparse
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
#  CONFIG  — tuned for RTX 4080S · Qwen2.5-7B · 1.3M rows
# ════════════════════════════════════════════════════════════════

INPUT_CSV      = "perfect_output.csv"
OUTPUT_CSV     = "enriched_output.csv"
DB_FILE        = "enricher.db"

VLLM_URL       = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_HEALTH    = "http://127.0.0.1:8000/v1/models"
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"

# KEY SPEED PARAM: 20 rows per LLM call = 20x fewer API calls
BATCH_SIZE     = 20
# RTX 4080S with 32GB: Qwen2.5-7B uses ~14GB → 18GB KV cache headroom
MAX_WORKERS    = 10
MAX_TOKENS     = 4096
TEMPERATURE    = 0.05     # near-deterministic → better JSON
MAX_RETRIES    = 3
SNAPSHOT_EVERY = 100      # export CSV every N completed batches

CB_THRESHOLD   = 5        # circuit breaker: consecutive failures before pause
CB_WAIT        = 20       # seconds to wait when circuit breaks


# ════════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ════════════════════════════════════════════════════════════════

OUT_COLUMNS = [
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

# Fields the LLM fills in (everything except raw_context, context, response, raw_response)
ANNOTATE_FIELDS = [
    "context_tone",
    "context_style",
    "context_emotion",
    "context_inner_thought",
    "response_tone",
    "response_style",
    "response_emotion",
    "response_inner_thought",
]


# ════════════════════════════════════════════════════════════════
#  PROMPTS
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You annotate dialogue pairs with tone, style, emotion, and inner thought.

INPUT: a JSON array of objects with fields: id, ctx (context), rsp (response)
OUTPUT: a JSON array of annotation objects — one per input, same order.

RETURN ONLY a valid JSON array. No markdown. No explanations.
Must start with [ and end with ].
Must have EXACTLY the same number of objects as the input.

Each output object schema:
{
  "id":   <same integer id as input>,
  "ct":   "context_tone1 + context_tone2",
  "cs":   "context_style1 + context_style2",
  "ce":   "context_emotion1 + context_emotion2",
  "ci":   "context inner thought if stated or strongly implied, else empty string",
  "rt":   "response_tone1 + response_tone2",
  "rs":   "response_style1 + response_style2",
  "re":   "response_emotion1 + response_emotion2",
  "ri":   "response inner thought if stated or strongly implied, else empty string"
}

TONE VALUES (combine with +):
formal | informal | casual | professional | friendly | warm | cold | distant |
polite | respectful | supportive | encouraging | comforting | gentle | soft |
serious | authoritative | confident | assertive | aggressive | hostile |
sarcastic | ironic | mocking | teasing | playful | humorous | witty |
romantic | flirtatious | affectionate | emotional | dramatic | passionate |
apologetic | sympathetic | empathetic | reassuring | persuasive | motivational |
questioning | curious | suspicious | defensive | awkward | hesitant | nervous |
excited | enthusiastic | energetic | calm | neutral

STYLE VALUES (combine with +):
dialogue | conversation | question | answer | statement | exclamation |
command | request | suggestion | advice | explanation | argument | complaint |
criticism | praise | encouragement | comfort | persuasion | confession |
apology | greeting | farewell | small_talk | banter | joke | humor |
sarcasm | teasing | flirting | romance | affection | inner_monologue |
observation | reaction | emotional_expression | character_thought | analysis

EMOTION VALUES (combine with +):
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

RULES:
- Output array must have EXACTLY as many objects as the input array.
- Use the id field — same integer from input.
- Never use null. Use empty string when data unavailable.
- inner_thought (ci/ri): only fill if the text explicitly states or strongly implies it.
- JSON validity is the top priority.
- A smaller valid array beats a larger invalid one.\
"""

REPAIR_SYSTEM = """\
Fix the broken JSON below. Return ONLY the corrected JSON array.
No markdown. No explanations. Start with [. End with ].\
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

def ok(m):   print(f"  {G}OK{RST}  {m}")
def warn(m): print(f"  {Y}!!{RST}  {m}")
def err(m):  print(f"  {R}XX{RST}  {m}")
def info(m): print(f"  {B}->{RST}  {m}")


# ════════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER
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

    def record_conn_fail(self):
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
        msg = f"vLLM unreachable — pausing {CB_WAIT}s..."
        (pbar.write if pbar else print)(Y + "  " + msg + RST)
        time.sleep(CB_WAIT)
        if _check_vllm():
            with self._lock:
                self._failures = 0
                self._open     = False
            (pbar.write if pbar else print)(G + "  vLLM back — resuming." + RST)


BREAKER = CircuitBreaker()


# ════════════════════════════════════════════════════════════════
#  VLLM HEALTH
# ════════════════════════════════════════════════════════════════

def _check_vllm() -> bool:
    try:
        r = requests.get(VLLM_HEALTH, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_vllm(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_vllm():
            return True
        time.sleep(5)
    return False


# ════════════════════════════════════════════════════════════════
#  DATABASE
# ════════════════════════════════════════════════════════════════

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
            id        INTEGER PRIMARY KEY,
            start_row INTEGER NOT NULL,
            end_row   INTEGER NOT NULL,
            status    TEXT    DEFAULT 'pending',
            attempts  INTEGER DEFAULT 0,
            error     TEXT    DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS results (
            row_id    INTEGER PRIMARY KEY,
            data      TEXT    NOT NULL,
            exported  INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_batches_status  ON batches(status);
        CREATE INDEX IF NOT EXISTS idx_results_exported ON results(exported);
    """)
    conn.commit()
    return conn


def reset_db(conn: sqlite3.Connection):
    conn.executescript("DELETE FROM results; DELETE FROM batches;")
    conn.commit()
    warn("Database cleared — starting fresh.")


def retry_failed(conn: sqlite3.Connection, lock: Lock):
    with lock:
        n = conn.execute(
            "UPDATE batches SET status='pending', attempts=0, error='' WHERE status='failed'"
        ).rowcount
        conn.commit()
    info(f"Reset {n} failed batches to pending.")


def populate_batches(conn: sqlite3.Connection, total_rows: int, batch_size: int) -> int:
    count = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    if count == 0:
        batches = [
            (i // batch_size, i, min(i + batch_size, total_rows))
            for i in range(0, total_rows, batch_size)
        ]
        conn.executemany(
            "INSERT INTO batches(id, start_row, end_row) VALUES (?, ?, ?)",
            batches,
        )
        conn.commit()
        ok(f"Created {len(batches)} batches for {total_rows:,} rows.")
        return len(batches)
    else:
        done    = conn.execute("SELECT COUNT(*) FROM batches WHERE status='done'").fetchone()[0]
        failed  = conn.execute("SELECT COUNT(*) FROM batches WHERE status='failed'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM batches WHERE status='pending'").fetchone()[0]
        info(f"Resuming: {done} done / {pending} pending / {failed} failed of {count} total.")
        return count


def get_pending_batches(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT id, start_row, end_row FROM batches "
        "WHERE status IN ('pending','failed') AND attempts < ?",
        (MAX_RETRIES,),
    ).fetchall()


def mark_batch(conn: sqlite3.Connection, lock: Lock,
               bid: int, status: str, error: str = ""):
    with lock:
        conn.execute(
            "UPDATE batches SET status=?, attempts=attempts+1, error=? WHERE id=?",
            (status, error[:300], bid),
        )
        conn.commit()


def store_results(conn: sqlite3.Connection, lock: Lock,
                  rows: list) -> int:
    """rows = list of (row_id, data_dict)"""
    stored = 0
    with lock:
        for row_id, data in rows:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO results(row_id, data, exported) VALUES (?, ?, 0)",
                    (row_id, json.dumps(data, ensure_ascii=False)),
                )
                stored += 1
            except Exception:
                pass
        conn.commit()
    return stored


def export_to_csv(conn: sqlite3.Connection, lock: Lock,
                  out_path: str, total_rows: int) -> int:
    with lock:
        rows = conn.execute(
            "SELECT row_id, data FROM results WHERE exported=0 ORDER BY row_id"
        ).fetchall()
    if not rows:
        return 0

    records, ids = [], []
    for row_id, data in rows:
        try:
            records.append(json.loads(data))
            ids.append(row_id)
        except Exception:
            pass
    if not records:
        return 0

    df = pd.DataFrame(records, columns=OUT_COLUMNS)
    file_exists = Path(out_path).exists()
    df.to_csv(out_path, mode="a", header=not file_exists, index=False, encoding="utf-8")

    with lock:
        conn.execute(
            "UPDATE results SET exported=1 WHERE row_id IN ({})".format(
                ",".join(["?"] * len(ids))
            ),
            ids,
        )
        conn.commit()
    return len(records)


# ════════════════════════════════════════════════════════════════
#  JSON EXTRACTION
# ════════════════════════════════════════════════════════════════

def _try_list(text: str) -> Optional[list]:
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else None
    except Exception:
        return None


def extract_json(text: str) -> Optional[list]:
    text = text.strip()

    # 1 Direct parse
    r = _try_list(text)
    if r is not None:
        return r

    # 2 Find [...] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        r = _try_list(m.group(0))
        if r is not None:
            return r

    # 3 Common fixes: trailing commas, single quotes
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    m = re.search(r"\[.*\]", fixed, re.DOTALL)
    if m:
        r = _try_list(m.group(0))
        if r is not None:
            return r

    # 4 Bracket-balance truncation
    try:
        start = text.index("[")
        depth, last_close = 0, -1
        in_str, esc = False, False
        for i, ch in enumerate(text[start:], start):
            if esc:        esc = False; continue
            if ch == "\\" and in_str: esc = True; continue
            if ch == '"':  in_str = not in_str; continue
            if in_str:     continue
            if ch == "[":  depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    last_close = i
                    break
        if last_close > start:
            r = _try_list(text[start:last_close + 1])
            if r is not None:
                return r
    except ValueError:
        pass

    # 5 Strip markdown and retry
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\[.*\]", stripped, re.DOTALL)
    if m:
        r = _try_list(m.group(0))
        if r is not None:
            return r

    return None


# ════════════════════════════════════════════════════════════════
#  vLLM CLIENT
# ════════════════════════════════════════════════════════════════

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json"})
        _tls.s = s
    return _tls.s


def call_llm(messages: list, temp: float = TEMPERATURE,
             max_tok: int = MAX_TOKENS) -> Optional[str]:
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "temperature": temp,
        "max_tokens":  max_tok,
    }
    for attempt in range(3):
        try:
            r = _session().post(VLLM_URL, json=payload, timeout=300)
            r.raise_for_status()
            BREAKER.record_success()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            BREAKER.record_conn_fail()
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            time.sleep(10 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            if code in (429, 503):
                time.sleep(15 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(5)
    return None


# ════════════════════════════════════════════════════════════════
#  ANNOTATION LOGIC
# ════════════════════════════════════════════════════════════════

# Short-key → long-key mapping (saves tokens in LLM output)
KEY_MAP = {
    "ct": "context_tone",
    "cs": "context_style",
    "ce": "context_emotion",
    "ci": "context_inner_thought",
    "rt": "response_tone",
    "rs": "response_style",
    "re": "response_emotion",
    "ri": "response_inner_thought",
}


def build_batch_payload(batch_rows: list) -> str:
    """Build compact JSON input for the LLM — short field names save tokens."""
    items = [
        {"id": r["_row_id"], "ctx": r["context"][:800], "rsp": r["response"][:400]}
        for r in batch_rows
    ]
    return json.dumps(items, ensure_ascii=False)


def parse_annotations(raw: list, batch_rows: list) -> dict:
    """
    Map LLM short-key output back to full column names.
    Returns {row_id: annotation_dict}.
    Handles out-of-order or missing IDs gracefully.
    """
    # Build id→row mapping
    id_to_row = {r["_row_id"]: r for r in batch_rows}
    result    = {}

    for item in raw:
        if not isinstance(item, dict):
            continue
        # Get the row id — could be int or str
        rid = item.get("id")
        if rid is None:
            continue
        try:
            rid = int(rid)
        except (ValueError, TypeError):
            continue

        if rid not in id_to_row:
            continue

        row = id_to_row[rid]
        ann = {
            "raw_context":   row["context"],
            "context":       row["context"],
            "response":      row["response"],
            "raw_response":  row["response"],
        }
        for short, long in KEY_MAP.items():
            ann[long] = str(item.get(short) or "")

        result[rid] = ann

    # Fill in any rows the LLM skipped (in case of partial output)
    for row in batch_rows:
        rid = row["_row_id"]
        if rid not in result:
            result[rid] = {
                "raw_context":            row["context"],
                "context":                row["context"],
                "response":               row["response"],
                "raw_response":           row["response"],
                "context_tone":           "",
                "context_style":          "",
                "context_emotion":        "",
                "context_inner_thought":  "",
                "response_tone":          "",
                "response_style":         "",
                "response_emotion":       "",
                "response_inner_thought": "",
            }

    return result


# ════════════════════════════════════════════════════════════════
#  BATCH PROCESSING  (runs inside thread pool)
# ════════════════════════════════════════════════════════════════

def process_batch(batch_id: int, start_row: int, end_row: int,
                  df: pd.DataFrame, conn: sqlite3.Connection,
                  db_lock: Lock, pbar=None) -> tuple:
    """Returns (rows_stored: int, status: str)"""

    BREAKER.wait_if_open(pbar)

    # Slice the shared DataFrame (read-only, no lock needed)
    slice_df   = df.iloc[start_row:end_row]
    batch_rows = []
    for idx, row in slice_df.iterrows():
        batch_rows.append({
            "_row_id":  idx,
            "context":  str(row.get("context", "") or ""),
            "response": str(row.get("response", "") or ""),
        })

    if not batch_rows:
        mark_batch(conn, db_lock, batch_id, "done")
        return 0, "done"

    payload_str = build_batch_payload(batch_rows)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": payload_str},
    ]

    # Pass 1: primary generation
    output = call_llm(messages)
    if not output:
        mark_batch(conn, db_lock, batch_id, "failed", "NO_LLM_RESPONSE")
        return 0, "failed"

    data = extract_json(output)

    # Pass 2: LLM repair
    if data is None:
        repair_out = call_llm(
            [{"role": "system", "content": REPAIR_SYSTEM},
             {"role": "user",   "content": output}],
            temp=0.0,
        )
        if repair_out:
            data = extract_json(repair_out)

    # Pass 3: strict retry
    if data is None:
        retry_out = call_llm(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": (
                 payload_str + "\n\nCRITICAL: Return ONLY a JSON array. "
                 "Start with [. End with ]. No other text."
             )}],
            temp=0.0,
        )
        if retry_out:
            data = extract_json(retry_out)

    if not data or not isinstance(data, list):
        mark_batch(conn, db_lock, batch_id, "failed", "INVALID_JSON_ALL_PASSES")
        return 0, "failed"

    # Parse and store annotations
    annotations = parse_annotations(data, batch_rows)
    rows_to_store = [
        (row_id, {col: ann.get(col, "") for col in OUT_COLUMNS})
        for row_id, ann in annotations.items()
    ]
    stored = store_results(conn, db_lock, rows_to_store)
    mark_batch(conn, db_lock, batch_id, "done")
    return stored, "done"


# ════════════════════════════════════════════════════════════════
#  STATS
# ════════════════════════════════════════════════════════════════

def print_stats(conn: sqlite3.Connection, elapsed: float, total_input: int):
    s = conn.execute("""
        SELECT
            SUM(CASE status WHEN 'done'    THEN 1 ELSE 0 END),
            SUM(CASE status WHEN 'failed'  THEN 1 ELSE 0 END),
            SUM(CASE status WHEN 'pending' THEN 1 ELSE 0 END),
            COUNT(*)
        FROM batches
    """).fetchone()
    done, failed, pending, total = [x or 0 for x in s]
    total_results  = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    exported_rows  = conn.execute("SELECT COUNT(*) FROM results WHERE exported=1").fetchone()[0]
    mins, secs     = divmod(int(elapsed), 60)

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  ENRICHMENT REPORT")
    print(sep)
    print(f"  Time elapsed          {mins}m {secs}s")
    print(f"  Input rows            {total_input:,}")
    print(f"  Batches done          {G}{done}{RST}")
    print(f"  Batches failed        {R}{failed}{RST}")
    print(f"  Batches pending       {Y}{pending}{RST}")
    print(f"  Rows annotated        {W}{total_results:,}{RST}")
    print(f"  Rows exported         {G}{exported_rows:,}{RST}")
    if done and elapsed:
        rows_per_sec = (done * BATCH_SIZE) / elapsed
        print(f"  Throughput            {rows_per_sec:.0f} rows/sec")
        remaining = (total - done) * BATCH_SIZE / max(rows_per_sec, 0.001)
        if remaining > 0 and pending > 0:
            rm, rs = divmod(int(remaining), 60)
            rh, rm = divmod(rm, 60)
            print(f"  Est. remaining        {rh}h {rm}m {rs}s")
    if total:
        pct = done / total * 100
        c = G if pct >= 90 else Y if pct >= 70 else R
        print(f"  Success rate          {c}{pct:.1f}%{RST}")
    print(sep + "\n")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CSV Enrichment Pipeline — adds tone/style/emotion to context+response CSV"
    )
    p.add_argument("--input",        type=str, default=INPUT_CSV,
                   help=f"Input CSV (default: {INPUT_CSV})")
    p.add_argument("--output",       type=str, default=OUTPUT_CSV,
                   help=f"Output CSV (default: {OUTPUT_CSV})")
    p.add_argument("--db",           type=str, default=DB_FILE,
                   help=f"SQLite DB file (default: {DB_FILE})")
    p.add_argument("--workers",      type=int, default=MAX_WORKERS,
                   help=f"Parallel workers (default {MAX_WORKERS})")
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE,
                   help=f"Rows per LLM call (default {BATCH_SIZE})")
    p.add_argument("--reset",        action="store_true",
                   help="Wipe DB and restart from scratch")
    p.add_argument("--retry-failed", action="store_true",
                   help="Reset all failed batches to pending and retry")
    p.add_argument("--export-only",  action="store_true",
                   help="Skip enrichment — just export DB rows to CSV")
    p.add_argument("--model", type=str, default=MODEL_NAME,
               help=f"vLLM model name (default: {MODEL_NAME})")
    p.add_argument("--url", type=str, default=VLLM_URL,
               help="vLLM API URL (default: localhost:8000)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    args       = parse_args()
    batch_size = args.batch_size
    sep        = "=" * 56
    MODEL_NAME = args.model
    VLLM_URL = args.url
    print(f"\n{sep}")
    print(f"  {W}CSV ENRICHMENT PIPELINE  TURBO{RST}")
    print(f"  Optimized for RTX 4080S  32 GB VRAM")
    print(sep + "\n")

    # DB
    conn    = init_db(args.db)
    db_lock = Lock()

    if args.reset:
        reset_db(conn)

    if args.retry_failed:
        retry_failed(conn, db_lock)

    # Export-only mode
    if args.export_only:
        info("Export-only mode.")
        n = export_to_csv(conn, db_lock, args.output, 0)
        ok(f"Exported {n:,} rows to {args.output}")
        print_stats(conn, 0, 0)
        conn.close()
        return

    # vLLM health check
    info(f"Checking vLLM at {VLLM_URL}...")
    if not _check_vllm():
        err("vLLM is NOT reachable!")
        print(f"""
  {Y}Start vLLM first in a SEPARATE terminal:{RST}

    python -m vllm.entrypoints.openai.api_server \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --max-model-len 8192 \\
      --gpu-memory-utilization 0.90 \\
      --max-num-seqs 32 \\
      --port 8000

  Waiting up to 2 minutes...
""")
        if not wait_for_vllm(120):
            err("vLLM still not reachable. Exiting.")
            sys.exit(1)
    ok("vLLM is online.\n")

    # Load CSV
    input_path = Path(args.input)
    if not input_path.exists():
        err(f"Input file not found: {args.input}")
        sys.exit(1)

    info(f"Loading {args.input}...")
    df = pd.read_csv(
        args.input,
        dtype=str,
        engine="python",
        on_bad_lines="skip"
    ).fillna("")

    # Must have context and response columns
    if "context" not in df.columns or "response" not in df.columns:
        err(f"CSV must have 'context' and 'response' columns. Found: {list(df.columns)}")
        sys.exit(1)

    total_rows = len(df)
    ok(f"Loaded {total_rows:,} rows.\n")

    # Create batches in DB
    populate_batches(conn, total_rows, batch_size)
    pending = get_pending_batches(conn)

    if not pending:
        warn("No pending batches. Use --reset or --retry-failed.")
        n = export_to_csv(conn, db_lock, args.output, total_rows)
        ok(f"Exported {n:,} rows to {args.output}")
        print_stats(conn, 0, total_rows)
        conn.close()
        return

    total_pending_rows = sum(end - start for _, start, end in pending)
    info(f"Batches pending:       {len(pending):,}")
    info(f"Rows pending:          {total_pending_rows:,}")
    info(f"Workers:               {args.workers}")
    info(f"Rows per LLM call:     {batch_size}  (= ~{total_pending_rows // batch_size:,} total API calls)")
    print()

    # Parallel enrichment
    start_time    = time.time()
    total_stored  = 0
    snap_counter  = 0
    fail_counter  = 0

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    with tqdm(total=len(pending), desc="Batches", unit="batch",
              ncols=88, bar_format=bar_fmt) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    process_batch, bid, start, end, df, conn, db_lock, pbar
                ): bid
                for bid, start, end in pending
            }
            for future in as_completed(future_map):
                bid = future_map[future]
                try:
                    stored, status = future.result()
                    total_stored += stored
                    snap_counter += 1
                    if status == "failed":
                        fail_counter += 1

                    pbar.set_postfix(
                        rows=f"{total_stored:,}",
                        fail=fail_counter,
                        s=status[0].upper(),
                        refresh=False,
                    )

                    # Snapshot export
                    if snap_counter >= SNAPSHOT_EVERY:
                        n = export_to_csv(conn, db_lock, args.output, total_rows)
                        snap_counter = 0
                        if n:
                            pbar.write(f"  Snapshot: {n:,} rows -> {args.output}")

                except Exception as e:
                    mark_batch(conn, db_lock, bid, "failed", str(e)[:200])
                    fail_counter += 1

                pbar.update(1)

    # Final export
    print()
    info("Final export...")
    n = export_to_csv(conn, db_lock, args.output, total_rows)
    ok(f"Exported {n:,} rows to {args.output}")

    elapsed = time.time() - start_time
    print_stats(conn, elapsed, total_rows)

    # Verify output
    if Path(args.output).exists():
        out_df = pd.read_csv(args.output)
        ok(f"Output file verified: {len(out_df):,} rows, {len(out_df.columns)} columns")
        info(f"Columns: {list(out_df.columns)}")

    conn.close()


if __name__ == "__main__":
    main()

'''
python -m pip install -U pip setuptools wheel
pip install -U vllm
pip install pandas requests tqdm orjson


vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384
'''