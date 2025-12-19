import os
import json
import time
import secrets
import re
from datetime import datetime, timezone
from urllib import request as urlrequest
from typing import Dict, Any, Optional

import psycopg2
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, Response
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "")
PREFILL_SECRET = os.getenv("PREFILL_SECRET", "")
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = os.getenv("DATABASE_URL", "")


def env_truthy(v: str) -> bool:
    return str(os.getenv(v, "")).strip().lower() in ("1", "true", "yes", "on")


ENABLE_PUBLISH = env_truthy("ENABLE_PUBLISH")
PUBLISH_UI_ENABLED = env_truthy("PUBLISH_UI_ENABLED")

GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "")
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "")
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "")

BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER", "")
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS", "")

_GBP_TOKEN_CACHE: Dict[str, Any] = {"token": None, "exp": 0}


# -------------------------------------------------
# Datenbank (prefill)
# -------------------------------------------------
def pg_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg2.connect(DATABASE_URL, sslmode="prefer", connect_timeout=5)


def prefill_init():
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS prefill (
                    rid TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at BIGINT NOT NULL,
                    used_at BIGINT,
                    used_count INT DEFAULT 0,
                    generated JSONB,
                    generated_at BIGINT
                )"""
            )
        conn.commit()


def prefill_insert(payload: dict) -> str:
    rid = secrets.token_urlsafe(18)
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prefill (rid,payload,created_at) VALUES (%s,%s,%s)",
                (rid, json.dumps(payload, ensure_ascii=False), int(time.time())),
            )
        conn.commit()
    return rid


def prefill_get_row(rid: str):
    if not rid:
        return None
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload,generated FROM prefill WHERE rid=%s", (rid,))
            row = cur.fetchone()
            if not row:
                return None
            payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            generated = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            return {"payload": payload, "generated": generated}


def prefill_set_generated(rid: str, data: dict):
    if not rid:
        return
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET generated=%s,generated_at=%s WHERE rid=%s",
                (json.dumps(data, ensure_ascii=False), int(time.time()), rid),
            )
        conn.commit()


# -------------------------------------------------
# Reviewer-Zeilen bereinigen
# -------------------------------------------------
def _suffix_line(name: str, date: str) -> str:
    parts = []
    if name.strip():
        parts.append(name.strip())
    if date.strip():
        parts.append(f"am {date.strip()}")
    if not parts:
        return ""
    return "— " + ", ".join(parts)


def _dedupe_reviewer(text: str, reviewer: str, reviewed_at: str) -> str:
    """
    Entfernt doppelte Zeilen wie
    — Rene Klein, am ...
    und hängt sie höchstens einmal an.
    """
    text = (text or "").strip()
    if not text:
        return text

    suf = _suffix_line(reviewer, reviewed_at)
    if not suf:
        return text

    # Normalize dash
    normalized_suffix = suf.replace("–", "—").replace("-", "—")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Entferne doppelte identische Suffix-Zeilen
    while lines and lines[-1].replace("–", "—").replace("-", "—") == normalized_suffix:
        lines.pop()

    lines.append(normalized_suffix)
    return "\n".join(lines)


# -------------------------------------------------
# Prefill API
# -------------------------------------------------
@app.post("/api/prefill")
def api_prefill():
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)
    data = request.get_json(force=True)

    review = (data.get("review") or "").strip()
    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()
    rating = str(data.get("rating") or "").strip()

    # Dedupe Reviewer-Zeilen
    review = _dedupe_reviewer(review, reviewer, reviewed_at)

    payload = {
        "review": review,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "rating": rating,
        "accountId": data.get("accountId"),
        "locationId": data.get("locationId"),
        "reviewId": data.get("reviewId"),
    }
    rid = prefill_insert(payload)
    return jsonify({"rid": rid})


# -------------------------------------------------
# Generate
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()
    reviews, replies = [{}], None

    if rid:
        row = prefill_get_row(rid)
        if row and row.get("payload"):
            p = row["payload"]
            reviews = [
                {
                    "review": _dedupe_reviewer(p.get("review", ""), p.get("reviewer", ""), p.get("reviewed_at", "")),
                    "rating": p.get("rating", ""),
                    "reviewType": "",
                    "salutation": "",
                }
            ]
            if row.get("generated"):
                replies = row["generated"].get("replies")

    return render_template(
        "index.html",
        reviews=reviews,
        replies=replies,
        rid=rid,
        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
    )


@app.post("/generate")
def generate():
    rid = (request.form.get("rid") or "").strip()
    reviews = request.form.getlist("review")
    ratings = request.form.getlist("rating")

    replies = []
    for idx, rev in enumerate(reviews[:MAX_REVIEWS]):
        if not rev.strip():
            continue
        rating = ratings[idx] if idx < len(ratings) else ""
        prompt = build_prompt(
            {
                "review": rev,
                "rating": rating,
                "selectedTone": request.form.get("selectedTone", "friendly"),
                "corporateSignature": request.form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
                "languageMode": request.form.get("languageMode", "de"),
            }
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        public, insights = split_public_and_insights(raw)
        replies.append({"review": rev, "reply": public, "insights": insights})

    if rid:
        prefill_set_generated(rid, {"replies": replies})
        return redirect(url_for("index", rid=rid))

    return render_template(
        "index.html",
        reviews=[{"review": r} for r in reviews],
        replies=replies,
        rid=rid,
        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
    )


if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
