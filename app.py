import os
import sys
import json
import time
import secrets
import re
from datetime import datetime, timezone
from urllib import request as urlrequest
from typing import Dict, Any, Optional, List

import psycopg2
from flask import Flask, render_template, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = (os.getenv("MAKE_WEBHOOK_URL") or "").strip()

PREFILL_SECRET = (os.getenv("PREFILL_SECRET") or "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

# --- Review-Normalisierung (verhindert doppelten Abbinder + entfernt "(Translated by Google)") ---
_TRANSLATION_MARKERS = [
    "\n\n(Translated by Google)\n",
    "\n\n(Übersetzt von Google)\n",
]

_RE_FOOTER_LINE = re.compile(r"^\s*—\s+.+$")


def _strip_translation_block(text: str) -> str:
    if not text:
        return text
    for marker in _TRANSLATION_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def _normalize_footer_block(lines: List[str]) -> List[str]:
    """
    If the review ends with 1+ footer lines starting with '—', keep only ONE footer line (the last one),
    and drop any additional footer lines (including duplicates).
    """
    if not lines:
        return lines

    # trim trailing empties
    while lines and not lines[-1].strip():
        lines.pop()

    # find trailing footer block
    i = len(lines) - 1
    footer_idxs = []
    while i >= 0 and _RE_FOOTER_LINE.match(lines[i].strip() or ""):
        footer_idxs.append(i)
        i -= 1

    if not footer_idxs:
        return lines

    footer_idxs.sort()
    last_footer = lines[footer_idxs[-1]].strip()

    # remove all footer lines, then append only the last one
    kept = [ln.rstrip() for idx, ln in enumerate(lines) if idx not in set(footer_idxs)]
    while kept and not kept[-1].strip():
        kept.pop()
    kept.append(last_footer)
    return kept


def normalize_review_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = _strip_translation_block(text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = _normalize_footer_block(lines)
    return "\n".join(lines).strip()


def _compose_review_text(review: str, reviewer: str = "", reviewed_at: str = "") -> str:
    """
    Produces: "<comment>\n— <reviewer>, am <reviewed_at>"
    But avoids adding the footer if it's already present (and dedupes if present multiple times).
    """
    review = normalize_review_text(review)

    reviewer = re.sub(r"\s+", " ", (reviewer or "").strip())
    reviewed_at = re.sub(r"\s+", " ", (reviewed_at or "").strip())

    if not review:
        return ""

    # If there is already a footer at the end, do NOT append a second one.
    lines = review.splitlines()
    if lines and _RE_FOOTER_LINE.match(lines[-1].strip() or ""):
        return review

    # No footer present; append only if we have data.
    if not reviewer and not reviewed_at:
        return review

    suffix_parts = []
    if reviewer:
        suffix_parts.append(reviewer)
    if reviewed_at:
        suffix_parts.append(f"am {reviewed_at}")

    suffix = ", ".join(suffix_parts)
    return f"{review}\n— {suffix}"


# --- DB ---
def pg_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing (add Railway Postgres)")
    return psycopg2.connect(
        DATABASE_URL,
        sslmode=os.getenv("PGSSLMODE", "prefer"),
        connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT", "5")),
    )


def prefill_init():
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prefill (
                  rid TEXT PRIMARY KEY,
                  payload JSONB NOT NULL,
                  created_at BIGINT NOT NULL,
                  used_at BIGINT,
                  used_count INT NOT NULL DEFAULT 0
                )
                """
            )
            # Backfill columns if table existed before
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS used_at BIGINT;")
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS used_count INT NOT NULL DEFAULT 0;")
        conn.commit()


def prefill_cleanup():
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE created_at < %s", (cutoff,))
        conn.commit()


def prefill_insert(payload: dict) -> str:
    rid = secrets.token_urlsafe(18)
    now = int(time.time())
    prefill_cleanup()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prefill (rid, payload, created_at, used_at, used_count) VALUES (%s, %s, %s, %s, %s)",
                (rid, json.dumps(payload, ensure_ascii=False), now, None, 0),
            )
        conn.commit()
    return rid


def prefill_get(rid: str):
    if not rid:
        return None
    prefill_cleanup()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM prefill WHERE rid = %s", (rid,))
            row = cur.fetchone()
            if not row:
                return None
            return row[0]


def prefill_mark_used(rid: str):
    if not rid:
        return
    now = int(time.time())
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET used_at = %s, used_count = used_count + 1 WHERE rid = %s",
                (now, rid),
            )
        conn.commit()


# --- API ---
@app.post("/api/prefill")
def api_prefill_create():
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(silent=True) or {}

    # Flexibel: akzeptiere "review" ODER (Make/GBP) "comment_full"/"comment"
    review_in = (data.get("review") or data.get("comment_full") or data.get("comment") or "").strip()
    rating_raw = data.get("rating")

    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()

    if not review_in or len(review_in) > 8000:
        return jsonify({"error": "invalid review"}), 400

    rating = ""
    if rating_raw is not None and str(rating_raw).strip() != "":
        try:
            r = int(str(rating_raw).strip())
            if r < 1 or r > 5:
                return jsonify({"error": "invalid rating"}), 400
            rating = str(r)
        except Exception:
            return jsonify({"error": "invalid rating"}), 400

    # Wichtig: hier entsteht der finale Text für das UI (ohne doppelten Abbinder)
    review_full = _compose_review_text(review_in, reviewer=reviewer, reviewed_at=reviewed_at)

    payload = {
        "review": review_full,
        "rating": rating,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
    }

    rid = prefill_insert(payload)
    resp = jsonify({"rid": rid})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()

    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }
    reviews = [{}]

    if rid:
        payload = prefill_get(rid)
        if payload:
            # zusätzliche Absicherung: alte/doppelte Daten beim Anzeigen normalisieren
            review_text = normalize_review_text(payload.get("review") or "")
            reviews = [{
                "review": review_text,
                "rating": payload.get("rating") or "",
                "reviewType": "",
                "salutation": "",
            }]

    return render_template("index.html", values=values, reviews=reviews, replies=None, rid=rid)


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form
    rid = (form.get("rid") or "").strip()
    if rid:
        # NICHT löschen, nur markieren
        try:
            prefill_mark_used(rid)
        except Exception as e:
            app.logger.warning("Prefill mark_used failed: %s", e)

    values = {
        "selectedTone": form.get("selectedTone", "friendly"),
        "corporateSignature": form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
        "contactEmail": form.get("contactEmail", "").strip(),
        "languageMode": form.get("languageMode", "de"),
    }

    reviews_list = [t.strip() for t in form.getlist("review")]
    ratings_list = form.getlist("rating")
    types_list = form.getlist("reviewType")
    sal_list = form.getlist("salutation")

    reviews_list = reviews_list[:MAX_REVIEWS]
    ratings_list = ratings_list[:MAX_REVIEWS]
    types_list = types_list[:MAX_REVIEWS]
    sal_list = sal_list[:MAX_REVIEWS]

    replies = []
    review_blocks = []

    base = {
        "selectedTone": values["selectedTone"],
        "languageMode": values["languageMode"],
        "corporateSignature": values["corporateSignature"],
        "contactEmail": values["contactEmail"],
    }

    for idx, review_text in enumerate(reviews_list):
        rating_raw = ratings_list[idx] if idx < len(ratings_list) else ""
        rtype = types_list[idx] if idx < len(types_list) else ""
        sal = sal_list[idx] if idx < len(sal_list) else ""

        # Normalisieren, damit auch manuell eingefügte doppel-Abbinder sauber werden
        review_text_clean = normalize_review_text(review_text)

        review_blocks.append(
            {"review": review_text_clean, "rating": rating_raw, "reviewType": rtype, "salutation": sal}
        )

        if not review_text_clean:
            continue

        data = dict(base)
        data["review"] = review_text_clean
        data["rating"] = rating_raw
        data["reviewType"] = rtype
        data["salutation"] = sal

        prompt = build_prompt(data)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        raw_reply = (response.choices[0].message.content or "").strip()
        public_answer, insights = split_public_and_insights(raw_reply)

        replies.append({"review": review_text_clean, "reply": public_answer, "insights": insights})

        if insights and MAKE_WEBHOOK_URL:
            try:
                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "review_text": review_text_clean,
                    "rating_input": rating_raw,
                    "tone": values["selectedTone"],
                    "language": values["languageMode"],
                    "insights": insights,
                }
                data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urlrequest.Request(
                    MAKE_WEBHOOK_URL,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urlrequest.urlopen(req, timeout=3)
            except Exception as e:
                app.logger.warning("Make webhook failed: %s", e)

    if not review_blocks:
        review_blocks = [{}]

    # rid im UI nicht weitertragen (optional). DB bleibt erhalten.
    return render_template("index.html", values=values, reviews=review_blocks, replies=replies, rid="")


# Tabelle beim Start sicherstellen
if DATABASE_URL:
    try:
        prefill_init()
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
