import os
import json
import time
import secrets
import hmac
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import psycopg2
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, abort, make_response
)
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

# --------------------------------------------------------
# Grundkonfiguration
# --------------------------------------------------------

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10

PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))

PUBLISH_PASSWORD = os.getenv("PUBLISH_PASSWORD", "").strip()

GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()

def env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")

ENABLE_PUBLISH = env_truthy("ENABLE_PUBLISH")
PUBLISH_UI_ENABLED = env_truthy("PUBLISH_UI_ENABLED")
PUBLISH_DRY_RUN = env_truthy("PUBLISH_DRY_RUN")

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def _utf8_len(s: str) -> int:
    return len((s or "").encode("utf-8"))

# --------------------------------------------------------
# Datenbank (Prefill)
# --------------------------------------------------------

def pg_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg2.connect(DATABASE_URL, sslmode="prefer", connect_timeout=5)

def prefill_init():
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prefill (
                    rid TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at BIGINT NOT NULL,
                    generated JSONB,
                    generated_at BIGINT
                )
            """)
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

def prefill_get_row(rid: str) -> Optional[dict]:
    if not rid:
        return None
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload,generated FROM prefill WHERE rid=%s",
                (rid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            payload_raw, generated_raw = row
            return {
                "payload": payload_raw,
                "generated": generated_raw,
            }

def prefill_set_generated(rid: str, data: dict):
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET generated=%s,generated_at=%s WHERE rid=%s",
                (json.dumps(data, ensure_ascii=False), int(time.time()), rid),
            )
        conn.commit()

# --------------------------------------------------------
# Index
# --------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()

    reviews = [{}]
    replies = None
    prefill_mode = bool(rid)
    location_title = ""

    if rid:
        row = prefill_get_row(rid)
        if row and row.get("payload"):
            p = row["payload"]
            location_title = p.get("locationTitle") or ""

            reviews = [{
                "review": p.get("review", ""),
                "rating": p.get("rating", ""),
                "reviewType": "",
                "salutation": "",
            }]

            if row.get("generated"):
                replies = (row["generated"] or {}).get("replies")

    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }

    return render_template(
        "index.html",
        values=values,
        reviews=reviews,
        replies=replies,
        rid=rid,
        prefill_mode=prefill_mode,
        location_title=location_title,
        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
        publish_dry_run=PUBLISH_DRY_RUN,
    )

# --------------------------------------------------------
# Generator
# --------------------------------------------------------

def _first_non_empty_pairs(reviews: List[str], ratings: List[str]):
    pairs = []
    for idx, rev in enumerate(reviews[:MAX_REVIEWS]):
        if (rev or "").strip():
            rat = ratings[idx] if idx < len(ratings) else ""
            pairs.append((rev.strip(), str(rat)))
    return pairs

@app.post("/generate")
def generate():
    rid = (request.form.get("rid") or "").strip()

    reviews      = request.form.getlist("review")
    ratings      = request.form.getlist("rating")
    salutations  = request.form.getlist("salutation")
    review_types = request.form.getlist("reviewType")

    values = {
        "selectedTone": request.form.get("selectedTone", "friendly"),
        "corporateSignature": request.form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
        "contactEmail": request.form.get("contactEmail", ""),
        "languageMode": request.form.get("languageMode", "de"),
    }

    pairs = _first_non_empty_pairs(reviews, ratings)

    if rid and pairs:
        pairs = [pairs[0]]

    replies_out = []

    for idx, (rev, rating) in enumerate(pairs):
        prompt = build_prompt({
            "review": rev,
            "rating": rating,
            "reviewType": review_types[idx] if idx < len(review_types) else "",
            "salutation": salutations[idx] if idx < len(salutations) else "",
            "selectedTone": values["selectedTone"],
            "corporateSignature": values["corporateSignature"],
            "contactEmail": values["contactEmail"],
            "languageMode": values["languageMode"],
        })

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        raw = (response.choices[0].message.content or "").strip()
        public, insights = split_public_and_insights(raw)

        replies_out.append({
            "review": rev,
            "reply": public,
            "insights": insights,
        })

    if rid:
        prefill_set_generated(rid, {"replies": replies_out})
        return redirect(url_for("index", rid=rid))

    return render_template(
        "index.html",
        values=values,
        reviews=[{"review": r} for r in reviews],
        replies=replies_out,
        rid="",
        prefill_mode=False,
        location_title="",
        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
        publish_dry_run=PUBLISH_DRY_RUN,
    )

# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
