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

PUBLISH_PASSWORD = os.getenv("PUBLISH_PASSWORD", "").strip()

GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()


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

def _json(data, status=200):
    return make_response(jsonify(data), status)

def _ensure_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


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
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload,generated,publish_result FROM prefill WHERE rid=%s",
                (rid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "payload": _ensure_dict(row[0]),
                "generated": _ensure_dict(row[1]),
                "publish_result": _ensure_dict(row[2]),
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

    payload = request.get_json(force=True) or {}
    rid = prefill_insert(payload)
    return jsonify({"rid": rid})


@app.get("/api/review-by-rid")
def api_review_by_rid():
    rid = (request.args.get("rid") or "").strip()
    row = prefill_get_row(rid)
    if not row:
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
# Google OAuth + Publish
# --------------------------------------------------------

def get_access_token() -> str:
    body = urlencode({
        "client_id": GBP_CLIENT_ID,
        "client_secret": GBP_CLIENT_SECRET,
        "refresh_token": GBP_REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }).encode("utf-8")

    req = Request(
        "https://oauth2.googleapis.com/token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]

def publish_reply(account_id, location_id, review_id, reply_text):
    access_token = get_access_token()
    url = f"https://mybusiness.googleapis.com/v4/accounts/{account_id}/locations/{location_id}/reviews/{review_id}/reply"

    body = json.dumps({"comment": reply_text}, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
    )

    with urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


@app.post("/api/publish")
def api_publish():
    if not _check_publish_password():
        return _json({"ok": False, "error": "unauthorized"}, 401)

    rid = (request.args.get("rid") or "").strip()
    body = request.get_json(silent=True) or {}
    reply_text = (body.get("reply") or "").strip()

    row = prefill_get_row(rid)
    if not row:
        return _json({"ok": False, "error": "rid not found"}, 404)

    p = row["payload"]

    if PUBLISH_DRY_RUN:
        prefill_set_published(rid, {"dry_run": True})
        return _json({"ok": True, "dry_run": True})

    result = publish_reply(
        p["accountId"],
        p["locationId"],
        p["reviewId"],
        reply_text,
    )

    prefill_set_published(rid, result)
    return _json({"ok": True, "result": result})


# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
