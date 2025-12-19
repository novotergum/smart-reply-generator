import os
import json
import time
import secrets
from datetime import datetime, timezone
from urllib import request as urlrequest
from typing import Dict, Any

import psycopg2
import requests
from flask import Flask, render_template, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Konfiguration
MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()
PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Publishing (Google Business Profile)
ENABLE_PUBLISH = os.getenv("ENABLE_PUBLISH", "").lower() in ["1", "true", "yes"]
GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()

# Optional Basic Auth
BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER", "").strip()
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS", "").strip()

# Einfaches Token-Caching
_GBP_TOKEN_CACHE: Dict[str, Any] = {"token": None, "exp": 0}


# --- Datenbank / Prefill ----------------------------------------------------------------

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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prefill (
                    rid TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at BIGINT NOT NULL,
                    used_at BIGINT,
                    used_count INT NOT NULL DEFAULT 0
                )
            """)
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
                """
                INSERT INTO prefill (rid, payload, created_at, used_at, used_count)
                VALUES (%s, %s, %s, NULL, 0)
                """,
                (rid, json.dumps(payload, ensure_ascii=False), now),
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
            return row[0] if row else None


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


# --- GBP API Helpers ----------------------------------------------------------------

def _publish_ready_from_payload(payload: dict) -> bool:
    if not payload:
        return False
    return bool(
        payload.get("accountId") and payload.get("locationId") and payload.get("reviewId")
    )


def _require_basic_auth_if_configured():
    if not (BASIC_AUTH_USER and BASIC_AUTH_PASS):
        return
    auth = request.authorization
    if not auth or auth.username != BASIC_AUTH_USER or auth.password != BASIC_AUTH_PASS:
        abort(401, description="Basic auth required")


def _gbp_access_token() -> str:
    if not (GBP_CLIENT_ID and GBP_CLIENT_SECRET and GBP_REFRESH_TOKEN):
        raise RuntimeError("GBP credentials missing")

    now = int(time.time())
    if _GBP_TOKEN_CACHE["token"] and _GBP_TOKEN_CACHE["exp"] - 60 > now:
        return _GBP_TOKEN_CACHE["token"]

    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": GBP_CLIENT_ID,
            "client_secret": GBP_CLIENT_SECRET,
            "refresh_token": GBP_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Token error {resp.status_code}: {resp.text}")

    j = resp.json()
    token = j.get("access_token", "").strip()
    expires_in = int(j.get("expires_in", 3600))

    if not token:
        raise RuntimeError(f"Token response missing access_token: {resp.text}")

    _GBP_TOKEN_CACHE.update({"token": token, "exp": now + expires_in})
    return token


def _gbp_get_review(name: str) -> dict:
    token = _gbp_access_token()
    url = f"https://mybusiness.googleapis.com/v4/{name}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"GBP get review failed {resp.status_code}: {resp.text}")
    return resp.json()


def _gbp_update_reply(name: str, reply_text: str) -> dict:
    token = _gbp_access_token()
    url = f"https://mybusiness.googleapis.com/v4/{name}/reply"
    body = {"comment": reply_text}
    resp = requests.put(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }, json=body, timeout=25)
    if resp.status_code >= 400:
        raise RuntimeError(f"GBP updateReply failed {resp.status_code}: {resp.text}")
    return resp.json()


# --- API Routes ----------------------------------------------------------------

@app.post("/api/prefill")
def api_prefill_create():
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(silent=True) or {}

    review = (data.get("review") or "").strip()
    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()

    account_id = (data.get("accountId") or "").strip()
    location_id = (data.get("locationId") or "").strip()
    review_id = (data.get("reviewId") or "").strip()
    store_code = (data.get("storeCode") or "").strip()
    location_title = (data.get("locationTitle") or "").strip()

    if not review or len(review) > 8000:
        return jsonify({"error": "invalid review"}), 400

    rating = ""
    rating_raw = data.get("rating")
    if rating_raw:
        try:
            r = int(str(rating_raw))
            if 1 <= r <= 5:
                rating = str(r)
            else:
                return jsonify({"error": "invalid rating"}), 400
        except Exception:
            return jsonify({"error": "invalid rating"}), 400

    review_full = review
    if reviewer or reviewed_at:
        suffix = ", ".join(filter(None, [reviewer, f"am {reviewed_at}" if reviewed_at else ""]))
        review_full = f"{review}\n— {suffix}"

    payload = {
        "review": review_full,
        "rating": rating,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "accountId": account_id,
        "locationId": location_id,
        "reviewId": review_id,
        "storeCode": store_code,
        "locationTitle": location_title,
    }

    rid = prefill_insert(payload)
    return jsonify({"rid": rid}), 200


@app.post("/api/publish_reply")
def api_publish_reply():
    if not ENABLE_PUBLISH:
        abort(404)
    _require_basic_auth_if_configured()

    data = request.get_json(silent=True) or {}
    rid = (data.get("rid") or "").strip()
    reply_text = (data.get("replyText") or "").strip()
    force = bool(data.get("force"))

    if not rid or not reply_text:
        return jsonify({"error": "missing rid/replyText"}), 400

    payload = prefill_get(rid)
    if not payload:
        return jsonify({"error": "rid not found or expired"}), 404

    account_id = payload.get("accountId", "").strip()
    location_id = payload.get("locationId", "").strip()
    review_id = payload.get("reviewId", "").strip()

    if not (account_id and location_id and review_id):
        return jsonify({"error": "rid payload missing accountId/locationId/reviewId"}), 400

    name = f"accounts/{account_id}/locations/{location_id}/reviews/{review_id}"

    if not force:
        try:
            review_obj = _gbp_get_review(name)
            existing = ((review_obj.get("reviewReply") or {}).get("comment") or "").strip()
            if existing:
                return jsonify({"error": "already_replied"}), 409
        except Exception as e:
            return jsonify({"error": "precheck_failed", "detail": str(e)}), 502

    try:
        rr = _gbp_update_reply(name, reply_text)
        return jsonify({"ok": True, "reviewReply": rr})
    except Exception as e:
        return jsonify({"error": "publish_failed", "detail": str(e)}), 502


# --- Hauptseiten ----------------------------------------------------------------

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
    publish_ready = False

    if rid:
        payload = prefill_get(rid)
        if payload:
            publish_ready = _publish_ready_from_payload(payload)
            reviews = [{
                "review": payload.get("review") or "",
                "rating": payload.get("rating") or "",
                "reviewType": "",
                "salutation": "",
            }]

    return render_template(
        "index.html",
        values=values,
        reviews=reviews,
        replies=None,
        rid=rid,
        publish_enabled=ENABLE_PUBLISH,
        publish_ready=publish_ready,
    )


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form
    rid = (form.get("rid") or "").strip()
    payload = prefill_get(rid) if rid else None
    publish_ready = _publish_ready_from_payload(payload) if payload else False

    if rid:
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

    reviews_list = [t.strip() for t in form.getlist("review")][:MAX_REVIEWS]
    ratings_list = form.getlist("rating")[:MAX_REVIEWS]
    types_list = form.getlist("reviewType")[:MAX_REVIEWS]
    sal_list = form.getlist("salutation")[:MAX_REVIEWS]

    replies = []
    review_blocks = []

    for idx, review_text in enumerate(reviews_list):
        if not review_text:
            continue

        rtype = types_list[idx] if idx < len(types_list) else ""
        rating_raw = ratings_list[idx] if idx < len(ratings_list) else ""
        sal = sal_list[idx] if idx < len(sal_list) else ""

        prompt_data = {
            "review": review_text,
            "rating": rating_raw,
            "reviewType": rtype,
            "salutation": sal,
            "selectedTone": values["selectedTone"],
            "corporateSignature": values["corporateSignature"],
            "contactEmail": values["contactEmail"],
            "languageMode": values["languageMode"],
        }

        prompt = build_prompt(prompt_data)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        raw_reply = (response.choices[0].message.content or "").strip()
        public_answer, insights = split_public_and_insights(raw_reply)

        # Doppelten Abbinder entfernen
        sig = values["corporateSignature"].strip()
        if sig and public_answer.endswith(sig):
            lines = public_answer.splitlines()
            if len(lines) > 1 and lines[-2].strip() == sig:
                lines = lines[:-1]
            public_answer = "\n".join(lines).strip()

        replies.append({"review": review_text, "reply": public_answer, "insights": insights})
        review_blocks.append({
            "review": review_text,
            "rating": rating_raw,
            "reviewType": rtype,
            "salutation": sal,
        })

        if insights and MAKE_WEBHOOK_URL:
            try:
                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "review_text": review_text,
                    "rating_input": rating_raw,
                    "tone": values["selectedTone"],
                    "language": values["languageMode"],
                    "insights": insights,
                }
                urlrequest.urlopen(urlrequest.Request(
                    MAKE_WEBHOOK_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ), timeout=3)
            except Exception as e:
                app.logger.warning("Webhook failed: %s", e)

    if not review_blocks:
        review_blocks = [{}]

    return render_template(
        "index.html",
        values=values,
        reviews=review_blocks,
        replies=replies,
        rid=rid,
        publish_enabled=ENABLE_PUBLISH,
        publish_ready=publish_ready and len(replies) == 1,
    )


# --- Start ----------------------------------------------------------------

if DATABASE_URL:
    try:
        prefill_init()
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
