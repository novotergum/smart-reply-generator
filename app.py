import os
import json
import time
import secrets
import hmac
from typing import Dict, Optional, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg2
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, abort, make_response
)
from flask_cors import CORS
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


# --------------------------------------------------------
# ✅ CORS – ABSOLUT KRITISCH
# --------------------------------------------------------
# ENV:
# ALLOWED_ORIGINS=https://ticket.novotergum.de
allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

CORS(
    app,
    resources={
        r"/api/review-by-rid": {"origins": allowed_origins},
        r"/api/prefill": {"origins": "*"},  # server-to-server
        r"/api/publish": {"origins": allowed_origins},
    },
    supports_credentials=False,
    methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "X-Prefill-Secret", "X-Publish-Password"],
)


# --------------------------------------------------------
# Feature Flags
# --------------------------------------------------------

def env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")

ENABLE_PUBLISH = env_truthy("ENABLE_PUBLISH")
PUBLISH_UI_ENABLED = env_truthy("PUBLISH_UI_ENABLED")
PUBLISH_DRY_RUN = env_truthy("PUBLISH_DRY_RUN")


# --------------------------------------------------------
# Defaults
# --------------------------------------------------------

def default_values() -> Dict[str, str]:
    return {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def _utf8_len(s: str) -> int:
    return len((s or "").encode("utf-8"))

def _check_publish_password() -> bool:
    if not PUBLISH_PASSWORD:
        return False
    candidate = (request.headers.get("X-Publish-Password") or "").strip()
    return hmac.compare_digest(candidate, PUBLISH_PASSWORD)

def must_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env: {name}")
    return v

def _json(data, status=200):
    return make_response(jsonify(data), status)

def _ensure_dict(v):
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return v


# --------------------------------------------------------
# Datenbank
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
                    generated_at BIGINT,
                    published_at BIGINT,
                    publish_result JSONB
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
                "SELECT payload,generated,created_at,published_at,publish_result "
                "FROM prefill WHERE rid=%s",
                (rid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            payload, generated, created_at, published_at, publish_result = row
            return {
                "payload": _ensure_dict(payload),
                "generated": _ensure_dict(generated),
                "created_at": created_at,
                "published_at": published_at,
                "publish_result": _ensure_dict(publish_result),
            }

def prefill_set_generated(rid: str, data: dict):
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET generated=%s,generated_at=%s WHERE rid=%s",
                (json.dumps(data, ensure_ascii=False), int(time.time()), rid),
            )
        conn.commit()

def prefill_set_published(rid: str, result: dict):
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET published_at=%s,publish_result=%s WHERE rid=%s",
                (int(time.time()), json.dumps(result, ensure_ascii=False), rid),
            )
        conn.commit()


# --------------------------------------------------------
# API: PREFILL
# --------------------------------------------------------

@app.post("/api/prefill")
def api_prefill():
    if (request.headers.get("X-Prefill-Secret") or "").strip() != PREFILL_SECRET:
        abort(401)

    data = request.get_json(force=True) or {}

    payload = {
        "review": data.get("review", ""),
        "rating": data.get("rating", ""),
        "reviewer": data.get("reviewer"),
        "reviewed_at": data.get("reviewed_at"),
        "accountId": data.get("accountId"),
        "locationId": data.get("locationId"),
        "reviewId": data.get("reviewId"),
        "storeCode": data.get("storeCode"),
        "locationTitle": data.get("locationTitle"),
        "maps_uri": data.get("maps_uri"),
        "new_review_uri": data.get("new_review_uri"),
        "place_id": data.get("place_id"),
        "maps_place_url": data.get("maps_place_url"),
    }

    rid = prefill_insert(payload)
    return jsonify({"rid": rid})


# --------------------------------------------------------
# ✅ API: REVIEW BY RID (Prefill-Endpunkt)
# --------------------------------------------------------

@app.get("/api/review-by-rid")
def api_review_by_rid():
    rid = (request.args.get("rid") or "").strip()
    if not rid:
        return jsonify({"error": "missing rid"}), 400

    row = prefill_get_row(rid)
    if not row or not row.get("payload"):
        return jsonify({"error": "not found"}), 404

    p = row["payload"]

    return jsonify({
        "review_text": p.get("review", ""),
        "rating": p.get("rating", ""),
        "reviewer": p.get("reviewer"),
        "reviewed_at": p.get("reviewed_at"),
        "locationTitle": p.get("locationTitle"),
    })


# --------------------------------------------------------
# Index
# --------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()
    reviews, replies = [{}], None
    prefill_mode = bool(rid)
    location_title = ""

    publish_ready = False
    publish_missing: List[str] = []
    google_check_url = None

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

            for k in ("accountId", "locationId", "reviewId"):
                if not p.get(k):
                    publish_missing.append(k)
            publish_ready = not publish_missing

            publish_result = row.get("publish_result") or {}
            google_check_url = publish_result.get("public_review_url")

    return render_template(
        "index.html",
        values=default_values(),
        reviews=reviews,
        replies=replies,
        rid=rid,
        prefill_mode=prefill_mode,
        location_title=location_title,
        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
        publish_ready=publish_ready,
        publish_missing=publish_missing,
        publish_dry_run=PUBLISH_DRY_RUN,
        google_check_url=google_check_url,
    )


# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
