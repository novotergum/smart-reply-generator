import os
import json
import time
import re
import secrets
from datetime import datetime, timezone
from urllib import request as urlrequest
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

import psycopg2
import requests
from flask import Flask, render_template, request, jsonify, abort, make_response
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()

PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


# Publishing (GBP)
ENABLE_PUBLISH = _env_bool("ENABLE_PUBLISH", False)
GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()

# Optional Basic Auth nur fürs Publishing (empfohlen)
BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER", "").strip()
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS", "").strip()

# simples Token-Caching (Refresh Token -> Access Token)
_GBP_TOKEN_CACHE: Dict[str, Any] = {"token": None, "exp": 0}


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
        conn.commit()


def prefill_cleanup():
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE created_at < %s", (cutoff,))
        conn.commit()


def _compose_review_text(review: str, reviewer: str = "", reviewed_at: str = "") -> str:
    """
    Baut optional eine Suffix-Zeile:
      — <reviewer>, am <reviewed_at>

    WICHTIG: Hängt NICHT erneut an, wenn im Text bereits eine identische Suffix-Zeile vorkommt
    (häufiger Datenfehler aus Upstream / E-Mail Copy&Paste).
    """
    review = (review or "").strip()
    reviewer = (reviewer or "").strip()
    reviewed_at = (reviewed_at or "").strip()

    if not reviewer and not reviewed_at:
        return review

    suffix_parts = []
    if reviewer:
        suffix_parts.append(reviewer)
    if reviewed_at:
        suffix_parts.append(f"am {reviewed_at}")

    suffix = ", ".join(suffix_parts).strip()
    if not suffix:
        return review

    suffix_line = f"— {suffix}".strip()

    # Wenn der Text bereits eine Zeile hat, die genau dieser Suffixline entspricht, nicht nochmals anhängen
    lines = [ln.strip() for ln in review.splitlines() if ln.strip()]
    if any(ln.strip() == suffix_line for ln in lines):
        return review

    return f"{review}\n{suffix_line}"


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


def _publish_ready_from_payload(payload: dict) -> bool:
    if not payload:
        return False
    return bool(
        (payload.get("accountId") or "").strip()
        and (payload.get("locationId") or "").strip()
        and (payload.get("reviewId") or "").strip()
    )


def _require_basic_auth_if_configured():
    if not (BASIC_AUTH_USER and BASIC_AUTH_PASS):
        return

    auth = request.authorization
    if auth and auth.username == BASIC_AUTH_USER and auth.password == BASIC_AUTH_PASS:
        return

    resp = make_response("Basic auth required", 401)
    resp.headers["WWW-Authenticate"] = 'Basic realm="Publish"'
    return resp


def _gbp_access_token() -> str:
    if not (GBP_CLIENT_ID and GBP_CLIENT_SECRET and GBP_REFRESH_TOKEN):
        raise RuntimeError("GBP env missing (GBP_CLIENT_ID/GBP_CLIENT_SECRET/GBP_REFRESH_TOKEN)")

    now = int(time.time())
    if _GBP_TOKEN_CACHE["token"] and (_GBP_TOKEN_CACHE["exp"] - 60) > now:
        return _GBP_TOKEN_CACHE["token"]

    url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": GBP_CLIENT_ID,
        "client_secret": GBP_CLIENT_SECRET,
        "refresh_token": GBP_REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }
    resp = requests.post(url, data=data, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"Token error {resp.status_code}: {resp.text}")

    j = resp.json()
    token = (j.get("access_token") or "").strip()
    expires_in = int(j.get("expires_in") or 3600)

    if not token:
        raise RuntimeError(f"Token response missing access_token: {resp.text}")

    _GBP_TOKEN_CACHE["token"] = token
    _GBP_TOKEN_CACHE["exp"] = now + expires_in
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
    resp = requests.put(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=25,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"GBP updateReply failed {resp.status_code}: {resp.text}")
    return resp.json()


def _extract_rid_fallback() -> str:
    """
    Falls das hidden input mal leer ist (Template/JS/Form-Änderungen),
    versuchen wir rid aus Referrer oder Query zu ziehen.
    """
    # 1) query
    rid = (request.args.get("rid") or "").strip()
    if rid:
        return rid

    # 2) referrer
    ref = (request.referrer or "").strip()
    if not ref:
        return ""

    try:
        q = parse_qs(urlparse(ref).query)
        rid2 = (q.get("rid", [""])[0] or "").strip()
        return rid2
    except Exception:
        return ""


def _normalize_signature_once(public_answer: str, signature: str) -> str:
    """
    Sorgt dafür, dass die Signatur am Ende maximal einmal erscheint.
    Entfernt doppelte/mehrfache Wiederholungen robust (Whitespace/Case/Leerzeilen).
    """
    text = (public_answer or "").strip()
    sig = (signature or "").strip()
    if not text or not sig:
        return text

    # Zeilenweise trailing cleanup
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and lines[-1].strip() == "":
        lines.pop()

    # Entferne beliebig viele Signaturzeilen am Ende (case-insensitive)
    while lines and lines[-1].strip().lower() == sig.lower():
        lines.pop()
        while lines and lines[-1].strip() == "":
            lines.pop()

    # Hänge genau einmal an
    if lines:
        return ("\n".join(lines).strip() + "\n\n" + sig).strip()
    return sig


@app.post("/api/prefill")
def api_prefill_create():
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(silent=True) or {}

    review = (data.get("review") or "").strip()
    rating_raw = data.get("rating")

    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()

    # GBP-Metadaten (für Publishing)
    account_id = (data.get("accountId") or "").strip()
    location_id = (data.get("locationId") or "").strip()
    review_id = (data.get("reviewId") or "").strip()
    store_code = (data.get("storeCode") or "").strip()
    location_title = (data.get("locationTitle") or "").strip()

    if not review or len(review) > 8000:
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

    review_full = _compose_review_text(review, reviewer=reviewer, reviewed_at=reviewed_at)

    payload = {
        # UI-Prefill
        "review": review_full,
        "rating": rating,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,

        # Publishing-Context
        "accountId": account_id,
        "locationId": location_id,
        "reviewId": review_id,
        "storeCode": store_code,
        "locationTitle": location_title,
    }

    rid = prefill_insert(payload)
    resp = jsonify({"rid": rid})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/publish_reply")
def api_publish_reply():
    if not ENABLE_PUBLISH:
        abort(404)

    auth_resp = _require_basic_auth_if_configured()
    if auth_resp:
        return auth_resp

    data = request.get_json(silent=True) or {}
    rid = (data.get("rid") or "").strip()
    reply_text = (data.get("replyText") or "").strip()
    force = bool(data.get("force"))

    if not rid or not reply_text:
        return jsonify({"error": "missing rid/replyText"}), 400
    if len(reply_text) > 4096:
        return jsonify({"error": "reply too long"}), 400

    payload = prefill_get(rid)
    if not payload:
        return jsonify({"error": "rid not found/expired"}), 404

    account_id = (payload.get("accountId") or "").strip()
    location_id = (payload.get("locationId") or "").strip()
    review_id = (payload.get("reviewId") or "").strip()

    if not (account_id and location_id and review_id):
        return jsonify({"error": "rid payload missing accountId/locationId/reviewId"}), 400

    name = f"accounts/{account_id}/locations/{location_id}/reviews/{review_id}"

    # Wenn nicht force: vorher prüfen, ob schon Reply existiert
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

    # Primär aus hidden input, fallback aus referrer/query, damit rid nicht "verschwindet"
    rid = (form.get("rid") or "").strip() or _extract_rid_fallback()

    payload = prefill_get(rid) if rid else None
    publish_ready = _publish_ready_from_payload(payload) if payload else False

    if rid and payload:
        try:
            prefill_mark_used(rid)
        except Exception as e:
            app.logger.warning("Prefill mark_used failed: %s", e)

    values = {
        "selectedTone": form.get("selectedTone", "friendly"),
        "corporateSignature": (form.get("corporateSignature", "Ihr NOVOTERGUM Team") or "").strip(),
        "contactEmail": (form.get("contactEmail", "") or "").strip(),
        "languageMode": form.get("languageMode", "de"),
    }

    reviews_list = [t.strip() for t in form.getlist("review")][:MAX_REVIEWS]
    ratings_list = form.getlist("rating")[:MAX_REVIEWS]
    types_list = form.getlist("reviewType")[:MAX_REVIEWS]
    sal_list = form.getlist("salutation")[:MAX_REVIEWS]

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

        review_blocks.append(
            {"review": review_text, "rating": rating_raw, "reviewType": rtype, "salutation": sal}
        )

        if not review_text:
            continue

        data = dict(base)
        data["review"] = review_text
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

        # Fix: Signatur am Ende nur einmal
        public_answer = _normalize_signature_once(public_answer, values["corporateSignature"])

        replies.append({"review": review_text, "reply": public_answer, "insights": insights})

        if insights and MAKE_WEBHOOK_URL:
            try:
                payload_out = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "review_text": review_text,
                    "rating_input": rating_raw,
                    "tone": values["selectedTone"],
                    "language": values["languageMode"],
                    "insights": insights,
                }
                data_bytes = json.dumps(payload_out).encode("utf-8")
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

    return render_template(
        "index.html",
        values=values,
        reviews=review_blocks,
        replies=replies,
        rid=rid,  # wichtig: nicht leeren
        publish_enabled=ENABLE_PUBLISH,
        publish_ready=bool(publish_ready and (len(replies) == 1)),
    )


# DB init (vorsichtig, aber mit connect_timeout)
if DATABASE_URL:
    try:
        prefill_init()
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
