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
# CORS (WICHTIG für ticket.novotergum.de -> Railway API)
# --------------------------------------------------------
# Beispiel:
# ALLOWED_ORIGINS=https://ticket.novotergum.de,https://ticket-staging.novotergum.de
allowed_origins = [
    o.strip() for o in (os.getenv("ALLOWED_ORIGINS", "")).split(",") if o.strip()
]

# Wenn du ALLOWED_ORIGINS nicht setzt, bleibt es dicht (sicherer Default)
cors_resources = {
    r"/api/review-by-rid": {"origins": allowed_origins},
    r"/api/prefill": {"origins": "*"},  # server-to-server, falls nötig
    r"/api/publish": {"origins": allowed_origins},
}

CORS(
    app,
    resources=cors_resources,
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
# Defaults (wichtig für Jinja)
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
    """psycopg2 kann JSONB je nach Setup als dict oder str liefern."""
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
# API: PREFILL  ✅ (Fix für Webhook / GitHub Actions)
# --------------------------------------------------------

@app.post("/api/prefill")
def api_prefill():
    if (request.headers.get("X-Prefill-Secret") or "").strip() != PREFILL_SECRET:
        abort(401)

    data = request.get_json(force=True) or {}

    payload = {
        "review": (data.get("review") or ""),
        "rating": (data.get("rating") or ""),
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
# ✅ API: REVIEW BY RID (für ticket.novotergum.de Prefill)
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
        # Optional: wenn du später die Form-Felder auto-füllen willst:
        "maps_place_url": p.get("maps_place_url"),
        "place_id": p.get("place_id"),
        "reviewId": p.get("reviewId"),
    })


# --------------------------------------------------------
# Google OAuth + Publish
# --------------------------------------------------------

def get_access_token() -> str:
    must_env("GBP_CLIENT_ID")
    must_env("GBP_CLIENT_SECRET")
    must_env("GBP_REFRESH_TOKEN")

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

    reviews = request.form.getlist("review")
    ratings = request.form.getlist("rating")
    salutations = request.form.getlist("salutation")
    review_types = request.form.getlist("reviewType")

    values = default_values()
    values.update({
        "selectedTone": request.form.get("selectedTone", values["selectedTone"]),
        "corporateSignature": request.form.get("corporateSignature", values["corporateSignature"]),
        "contactEmail": request.form.get("contactEmail", ""),
    })

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
        publish_ready=False,
        publish_missing=["accountId", "locationId", "reviewId"],
        publish_dry_run=PUBLISH_DRY_RUN,
        google_check_url=None,
    )


# --------------------------------------------------------
# API: Publish + Google-Link
# --------------------------------------------------------

@app.post("/api/publish")
def api_publish():
    if not _check_publish_password():
        return _json({"ok": False, "error": "unauthorized"}, 401)

    if not ENABLE_PUBLISH:
        return _json({"ok": False, "error": "publishing disabled"}, 403)

    rid = (request.args.get("rid") or "").strip()
    body = request.get_json(silent=True) or {}
    reply_text = (body.get("reply") or "").strip()

    if not rid:
        return _json({"ok": False, "error": "missing rid"}, 400)

    row = prefill_get_row(rid)
    if not row or not row.get("payload"):
        return _json({"ok": False, "error": "rid not found"}, 404)

    payload = row["payload"]

    for k in ("accountId", "locationId", "reviewId"):
        if not payload.get(k):
            return _json({"ok": False, "error": "publish not ready", "missing": k}, 400)

    if not reply_text:
        return _json({"ok": False, "error": "no reply text"}, 400)

    if _utf8_len(reply_text) > 4096:
        return _json({"ok": False, "error": "reply too long"}, 400)

    public_review_url = None
    if payload.get("maps_place_url") and payload.get("reviewId"):
        public_review_url = f'{payload["maps_place_url"]}&reviewId={payload["reviewId"]}'
    elif payload.get("place_id") and payload.get("reviewId"):
        public_review_url = (
            "https://www.google.com/maps/place/?q=place_id="
            f'{payload["place_id"]}&reviewId={payload["reviewId"]}'
        )

    if PUBLISH_DRY_RUN:
        prefill_set_published(rid, {
            "dry_run": True,
            "public_review_url": public_review_url
        })
        return _json({"ok": True, "dry_run": True, "public_review_url": public_review_url})

    try:
        result = publish_reply(
            payload["accountId"],
            payload["locationId"],
            payload["reviewId"],
            reply_text,
        )
        prefill_set_published(rid, {
            "dry_run": False,
            "result": result,
            "public_review_url": public_review_url,
        })
        return _json({
            "ok": True,
            "result": result,
            "public_review_url": public_review_url,
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
