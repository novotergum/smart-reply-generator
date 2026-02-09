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
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))

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
# Helpers
# --------------------------------------------------------

def _corsify(resp):
    origin = request.headers.get("Origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Publish-Password"
    return resp

def _json(data, status=200):
    return _corsify(make_response(jsonify(data), status))

def _check_publish_password() -> bool:
    if not PUBLISH_PASSWORD:
        return False
    candidate = (request.headers.get("X-Publish-Password") or "").strip()
    return hmac.compare_digest(candidate, PUBLISH_PASSWORD)

def _utf8_len(s: str) -> int:
    return len((s or "").encode("utf-8"))

def compute_publish_ready(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing = []
    for k in ("accountId", "locationId", "reviewId"):
        if not str(payload.get(k) or "").strip():
            missing.append(k)
    return (len(missing) == 0), missing

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
                "SELECT payload,generated,created_at,published_at,publish_result FROM prefill WHERE rid=%s",
                (rid,),
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
                "publish_result": publish_result,
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
        raw = resp.read().decode("utf-8")
    return json.loads(raw)["access_token"]

def publish_reply(account_id: str, location_id: str, review_id: str, reply_text: str) -> dict:
    access_token = get_access_token()
    name = f"accounts/{account_id}/locations/{location_id}/reviews/{review_id}"
    url = f"https://mybusiness.googleapis.com/v4/{name}/reply"

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
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {"ok": True}

# --------------------------------------------------------
# API: Publish (FIX)
# --------------------------------------------------------

@app.route("/api/publish", methods=["POST", "OPTIONS"])
def api_publish():
    if request.method == "OPTIONS":
        return _corsify(make_response("", 204))

    if not ENABLE_PUBLISH:
        return _json({"ok": False, "error": "publishing disabled"}, 403)

    if not _check_publish_password():
        return _json({"ok": False, "error": "unauthorized"}, 401)

    rid = (request.args.get("rid") or "").strip()
    body = request.get_json(silent=True) or {}
    reply = (body.get("reply") or "").strip()

    row = prefill_get_row(rid)
    if not row:
        return _json({"ok": False, "error": "rid not found"}, 404)

    created_at = int(row.get("created_at") or 0)
    if time.time() > created_at + PREFILL_TTL_SECONDS:
        return _json({"ok": False, "error": "rid expired", "redirect": "/"}, 410)

    payload = row["payload"]
    ready, missing = compute_publish_ready(payload)
    if not ready:
        return _json({"ok": False, "error": "publish not ready", "missing": missing}, 400)

    if _utf8_len(reply) > 4096:
        return _json({"ok": False, "error": "reply too long"}, 400)

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
# Index + Generator (unverändert)
# --------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()
    reviews, replies = [{}], None
    prefill_mode = bool(rid)
    location_title = ""

    publish_ready = False
    publish_missing = []

    if rid:
        row = prefill_get_row(rid)
        if row:
            p = row["payload"]
            location_title = p.get("locationTitle") or ""
            reviews = [{"review": p.get("review", ""), "rating": p.get("rating", "")}]
            if row.get("generated"):
                replies = row["generated"].get("replies")
            publish_ready, publish_missing = compute_publish_ready(p)

    return render_template(
        "index.html",
        values={
            "selectedTone": "friendly",
            "corporateSignature": "Ihr NOVOTERGUM Team",
            "contactEmail": "",
            "languageMode": "de",
        },
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
    )

@app.post("/generate")
def generate():
    reviews = request.form.getlist("review")
    ratings = request.form.getlist("rating")
    pairs = [(r, ratings[i] if i < len(ratings) else "") for i, r in enumerate(reviews) if r.strip()]

    replies_out = []
    for rev, rating in pairs:
        prompt = build_prompt({"review": rev, "rating": rating})
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        public, insights = split_public_and_insights(raw)
        replies_out.append({"review": rev, "reply": public, "insights": insights})

    return render_template("index.html", reviews=reviews, replies=replies_out)

# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
