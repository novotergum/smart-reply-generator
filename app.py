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
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, make_response
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
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "1209600"))  # 14 Tage

PUBLISH_PASSWORD = os.getenv("PUBLISH_PASSWORD", "").strip()

GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()

def env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")

ENABLE_PUBLISH = env_truthy("ENABLE_PUBLISH")
PUBLISH_UI_ENABLED = env_truthy("PUBLISH_UI_ENABLED")
PUBLISH_DRY_RUN = env_truthy("PUBLISH_DRY_RUN")

PUBLISH_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("PUBLISH_ALLOWED_ORIGINS", "").split(",") if o.strip()
]

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def _corsify(resp):
    origin = request.headers.get("Origin", "")
    if PUBLISH_ALLOWED_ORIGINS:
        if origin in PUBLISH_ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Originorigin"] = origin
            resp.headers["Vary"] = "Origin"
    elif origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"

    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Prefill-Secret, X-Publish-Password"
    return resp

def _json(data, status=200):
    return _corsify(make_response(jsonify(data), status))

def _check_publish_password() -> bool:
    if not PUBLISH_PASSWORD:
        return False
    candidate = (request.headers.get("X-Publish-Password") or "").strip()
    return hmac.compare_digest(candidate, PUBLISH_PASSWORD)

# --------------------------------------------------------
# Datenbank
# --------------------------------------------------------

def pg_connect():
    return psycopg2.connect(DATABASE_URL, sslmode="prefer", connect_timeout=5)

def prefill_init():
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prefill (
                    rid TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at BIGINT NOT NULL,
                    used_at BIGINT,
                    used_count INT DEFAULT 0,
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
                (rid, json.dumps(payload, ensure_ascii=False), int(time.time()))
            )
        conn.commit()
    return rid

def prefill_get_row(rid: str) -> Optional[dict]:
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload, generated, created_at, published_at, publish_result FROM prefill WHERE rid=%s",
                (rid,)
            )
            row = cur.fetchone()
            if not row:
                return None

            payload, generated, created_at, published_at, publish_result = row
            return {
                "payload": payload,
                "generated": generated,
                "created_at": created_at,
                "published_at": published_at,
                "publish_result": publish_result
            }

def prefill_set_generated(rid: str, data: dict):
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET generated=%s, generated_at=%s WHERE rid=%s",
                (json.dumps(data, ensure_ascii=False), int(time.time()), rid)
            )
        conn.commit()

def prefill_set_published(rid: str, result: dict):
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET published_at=%s, publish_result=%s WHERE rid=%s",
                (int(time.time()), json.dumps(result, ensure_ascii=False), rid)
            )
        conn.commit()

# --------------------------------------------------------
# 🔑 ZENTRALER RID-CHECK (FIX)
# --------------------------------------------------------

def check_rid_valid(row: dict):
    created_at = int(row.get("created_at") or 0)
    if not created_at:
        return False, "rid invalid"

    expires_at = created_at + PREFILL_TTL_SECONDS
    now = int(time.time())

    if now > expires_at:
        return False, "rid expired"

    return True, None

# --------------------------------------------------------
# API
# --------------------------------------------------------

@app.post("/api/prefill")
def api_prefill():
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(force=True) or {}

    payload = {
        "review": data.get("review", ""),
        "reviewer": data.get("reviewer", ""),
        "reviewed_at": data.get("reviewed_at", ""),
        "rating": str(data.get("rating") or ""),
        "accountId": data.get("accountId", ""),
        "locationId": data.get("locationId", ""),
        "reviewId": data.get("reviewId", ""),
        "storeCode": data.get("storeCode"),
        "locationTitle": data.get("locationTitle"),
        "maps_uri": data.get("maps_uri"),
        "new_review_uri": data.get("new_review_uri"),
        "place_id": data.get("place_id"),
        "maps_place_url": data.get("maps_place_url"),
    }

    rid = prefill_insert(payload)
    return jsonify({"rid": rid})

@app.route("/api/review-by-rid", methods=["GET", "OPTIONS"])
def api_review_by_rid():
    if request.method == "OPTIONS":
        return _corsify(make_response("", 204))

    rid = (request.args.get("rid") or "").strip()
    row = prefill_get_row(rid)
    if not row:
        return _json({"error": "rid not found"}, 404)

    ok, err = check_rid_valid(row)
    if not ok:
        return _json({"error": err}, 410)

    payload = row["payload"]
    return _json({
        "rid": rid,
        "review_text": payload.get("review"),
        "rating": payload.get("rating"),
        "reviewer": payload.get("reviewer"),
        "reviewed_at": payload.get("reviewed_at"),
    })

@app.route("/api/publish", methods=["POST", "OPTIONS"])
def api_publish():
    if request.method == "OPTIONS":
        return _corsify(make_response("", 204))

    if not _check_publish_password():
        return _json({"error": "unauthorized"}, 401)

    rid = request.args.get("rid") or (request.get_json() or {}).get("rid")
    row = prefill_get_row(rid)
    if not row:
        return _json({"error": "rid not found"}, 404)

    ok, err = check_rid_valid(row)
    if not ok:
        return _json({"error": err}, 410)

    payload = row["payload"]
    reply = (request.get_json() or {}).get("reply", "").strip()

    if not reply:
        generated = row.get("generated") or {}
        replies = generated.get("replies") or []
        if replies:
            reply = replies[0].get("reply", "")

    if not reply:
        return _json({"error": "no reply text"}, 400)

    if PUBLISH_DRY_RUN:
        prefill_set_published(rid, {"dry_run": True})
        return _json({"ok": True, "dry_run": True})

    result = publish_reply(
        payload["accountId"],
        payload["locationId"],
        payload["reviewId"],
        reply
    )

    prefill_set_published(rid, result)
    return _json({"ok": True, "result": result})

# --------------------------------------------------------
# UI
# --------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    rid = request.args.get("rid")
    row = prefill_get_row(rid) if rid else None

    reviews = []
    replies = None

    if row:
        payload = row["payload"]
        reviews = [{
            "review": payload.get("review"),
            "rating": payload.get("rating"),
        }]
        if row.get("generated"):
            replies = row["generated"].get("replies")

    return render_template(
        "index.html",
        reviews=reviews or [{}],
        replies=replies,
        rid=rid,
        prefill_mode=bool(rid),
        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
        publish_dry_run=PUBLISH_DRY_RUN,
        published_at=row.get("published_at") if row else None,
        publish_result=row.get("publish_result") if row else None,
    )

# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
