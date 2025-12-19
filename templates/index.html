import os
import json
import time
import secrets
import re
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

MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()

PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Publishing (GBP)
def _env_truthy(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

ENABLE_PUBLISH = _env_truthy("ENABLE_PUBLISH", "0")
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
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS used_at BIGINT;")
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS used_count INT NOT NULL DEFAULT 0;")
        conn.commit()


def prefill_cleanup():
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE created_at < %s", (cutoff,))
        conn.commit()


def _dedupe_consecutive_lines(lines):
    out = []
    for l in lines:
        if out and l == out[-1]:
            continue
        out.append(l)
    return out


def _compose_review_text(review: str, reviewer: str = "", reviewed_at: str = "") -> str:
    """
    Idempotent:
    - entfernt doppelte Abbinder-Zeilen am Ende (z. B. zweimal "— Rene Klein, am ...")
    - hängt Abbinder nur an, wenn er noch nicht vorhanden ist
    """
    review = (review or "").strip()
    reviewer = (reviewer or "").strip()
    reviewed_at = (reviewed_at or "").strip()

    if not review:
        return ""

    # Normalize lines + remove consecutive duplicates (hilft gegen 2x gleiche Zeile)
    lines = [ln.rstrip() for ln in review.splitlines()]
    lines = _dedupe_consecutive_lines(lines)

    # Entferne doppelte Abbinder-Zeilen am Ende auch wenn nicht direkt aufeinander folgen
    # (typisch: letzter und vorletzter identisch)
    while len(lines) >= 2 and lines[-1].strip() and lines[-1].strip() == lines[-2].strip():
        lines.pop()

    # Wenn kein Reviewer/Datum geliefert wird: nur dedupe anwenden
    if not reviewer and not reviewed_at:
        return "\n".join(lines).strip()

    # Ziel-Abbinder in einheitlicher Form
    suffix_parts = []
    if reviewer:
        suffix_parts.append(reviewer)
    if reviewed_at:
        suffix_parts.append(f"am {reviewed_at}")
    suffix = ", ".join(suffix_parts)
    suffix_line = f"— {suffix}".strip()

    # Prüfen: existiert bereits ein Abbinder mit Reviewer/Datum in den letzten 3 Zeilen?
    tail = [l.strip() for l in lines[-3:]] if len(lines) >= 3 else [l.strip() for l in lines]

    if suffix_line in tail or suffix_line in lines:
        return "\n".join(lines).strip()

    # Fuzzy-Erkennung: "— <reviewer> ... <date>"
    if reviewer:
        pat = r"^\s*—\s*.*" + re.escape(reviewer) + r".*"
        if reviewed_at:
            pat += re.escape(reviewed_at) + r".*"
        rx = re.compile(pat, re.IGNORECASE)
        if any(rx.match(l) for l in tail):
            return "\n".join(lines).strip()

    # Ansonsten anhängen
    return ("\n".join(lines).strip() + "\n" + suffix_line).strip()


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
    if not auth or auth.username != BASIC_AUTH_USER or auth.password != BASIC_AUTH_PASS:
        return abort(401, description="Basic auth required")


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


def _dedupe_trailing_signature(text: str, signature: str) -> str:
    """
    Falls das Modell die Signatur doppelt ausgibt, entfernen wir Duplikate am Ende.
    """
    t = (text or "").strip()
    sig = (signature or "").strip()
    if not t or not sig:
        return t

    lines = [ln.rstrip() for ln in t.splitlines()]
    # Entferne leere Endzeilen
    while lines and not lines[-1].strip():
        lines.pop()

    # Entferne doppelte Signatur am Ende
    while len(lines) >= 2 and lines[-1].strip() == sig and lines[-2].strip() == sig:
        lines.pop()

    return "\n".join(lines).strip()


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

    # WICHTIG: idempotent (verhindert doppelten Reviewer/Datum-Abbinder)
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

    _require_basic_auth_if_configured()

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
        "contactEmail": (form.get("contactEmail", "") or "").strip(),
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

        # Schutz gegen doppelte Signatur am Ende (falls Prompt+Model doppelt)
        public_answer = _dedupe_trailing_signature(public_answer, values["corporateSignature"])

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

    # rid NICHT leeren
    return render_template(
        "index.html",
        values=values,
        reviews=review_blocks,
        replies=replies,
        rid=rid,
        publish_enabled=ENABLE_PUBLISH,
        publish_ready=publish_ready and (len(replies) == 1),
    )


if DATABASE_URL:
    try:
        prefill_init()
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
