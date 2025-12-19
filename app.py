import os
import sys
import json
import time
import secrets
import re
from datetime import datetime, timezone
from urllib import request as urlrequest

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

# Google Business Profile / My Business v4 (Reviews Reply)
GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()

# Optional: Basic-Auth zum Absichern des Publish-Buttons
BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER", "").strip()
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS", "").strip()

# Optional: Entfernt "(Translated by Google)"-Block
STRIP_TRANSLATED_BLOCK = (os.getenv("STRIP_TRANSLATED_BLOCK", "1").strip() == "1")

# In-memory Token Cache (pro Worker)
_GBP_TOKEN = {"access_token": None, "exp": 0}

PUBLIC_APP_URL = os.getenv("PUBLIC_APP_URL", "").strip() or "https://smart-reply-generator-production2.up.railway.app"


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


def _strip_translated_block(text: str) -> str:
    """
    Entfernt typische Google-UI Übersetzungsblöcke.
    Beispiel:
      <original>
      (Translated by Google)
      <translation>
    -> behält nur <original>
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()

    if not STRIP_TRANSLATED_BLOCK:
        return t

    markers = [
        "\n(Translated by Google)",
        "\n(Übersetzt von Google)",
        "\n(Translated by Google)\n",
        "\n(Übersetzt von Google)\n",
    ]
    idx = -1
    for m in markers:
        j = t.find(m)
        if j != -1:
            idx = j
            break
    if idx != -1:
        t = t[:idx].rstrip()

    return t


def _dedupe_trailing_attribution_lines(text: str) -> str:
    """
    Entfernt doppelte Attribution-Zeilen am Ende, z.B.
      — Name, am ...
      — Name, am ...
    """
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").split("\n")

    # trim trailing empties
    while lines and not lines[-1].strip():
        lines.pop()

    # remove consecutive duplicate attribution lines at end
    while len(lines) >= 2:
        a = lines[-1].strip()
        b = lines[-2].strip()
        if a == b and a.startswith("—"):
            lines.pop()
        else:
            break

    return "\n".join(lines).strip()


def _compose_review_text(review: str, reviewer: str = "", reviewed_at: str = "") -> str:
    """
    Baut "review_full" wie im Mail-Format:
      <review>
      — <reviewer>, am <reviewed_at>
    Idempotent: falls am Ende bereits eine '— ...' Zeile existiert, wird NICHT erneut angehängt.
    """
    review = _strip_translated_block(review or "")
    review = _dedupe_trailing_attribution_lines(review)

    review = (review or "").strip()
    reviewer = (reviewer or "").strip()
    reviewed_at = (reviewed_at or "").strip()

    if not review:
        return ""

    # Wenn bereits eine Attribution-Zeile existiert, nicht erneut anhängen
    # (wir prüfen nur die letzte nicht-leere Zeile)
    last_line = review.split("\n")[-1].strip()
    if last_line.startswith("—"):
        return review

    if not reviewer and not reviewed_at:
        return review

    suffix_parts = []
    if reviewer:
        suffix_parts.append(reviewer)
    if reviewed_at:
        suffix_parts.append(f"am {reviewed_at}")

    suffix = ", ".join(suffix_parts)
    return f"{review}\n— {suffix}"


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


def _basic_auth_required() -> bool:
    return bool(BASIC_AUTH_USER and BASIC_AUTH_PASS)


def _check_basic_auth():
    if not _basic_auth_required():
        return
    auth = request.authorization
    if not auth or auth.username != BASIC_AUTH_USER or auth.password != BASIC_AUTH_PASS:
        # Browser-Prompt
        return abort(401, {"WWW-Authenticate": 'Basic realm="Smart Reply"'})
    return


def _gbp_enabled() -> bool:
    return bool(GBP_CLIENT_ID and GBP_CLIENT_SECRET and GBP_REFRESH_TOKEN)


def gbp_access_token() -> str:
    if not _gbp_enabled():
        raise RuntimeError("GBP OAuth env missing (GBP_CLIENT_ID/SECRET/REFRESH_TOKEN)")

    now = int(time.time())
    if _GBP_TOKEN["access_token"] and now < (_GBP_TOKEN["exp"] - 60):
        return _GBP_TOKEN["access_token"]

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
    token = j.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token: {resp.text}")

    exp = now + int(j.get("expires_in", 3600))
    _GBP_TOKEN.update({"access_token": token, "exp": exp})
    return token


def gbp_get_review(account_id: str, location_id: str, review_id: str) -> dict:
    token = gbp_access_token()
    url = f"https://mybusiness.googleapis.com/v4/accounts/{account_id}/locations/{location_id}/reviews/{review_id}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"GET review failed {resp.status_code}: {resp.text}")
    return resp.json()


def gbp_update_reply(account_id: str, location_id: str, review_id: str, reply_text: str) -> dict:
    token = gbp_access_token()
    name = f"accounts/{account_id}/locations/{location_id}/reviews/{review_id}"
    # Laut Doku: PUT .../v4/{name=accounts/*/locations/*/reviews/*}/reply :contentReference[oaicite:2]{index=2}
    url = f"https://mybusiness.googleapis.com/v4/{name}/reply"
    body = {"comment": reply_text}
    resp = requests.put(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=body, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"updateReply failed {resp.status_code}: {resp.text}")
    return resp.json()


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

    # Optional metadata for publishing reply
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
        "review": review_full,
        "rating": rating,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        # meta (optional)
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
    """
    POST JSON:
      { "rid": "...", "replyText": "...", "force": false }
    - Lädt accountId/locationId/reviewId aus Prefill payload
    - Blockt Überschreiben, wenn bereits eine Antwort existiert (außer force=true)
    """
    _check_basic_auth()

    if not _gbp_enabled():
        return jsonify({"error": "GBP OAuth not configured"}), 500

    data = request.get_json(silent=True) or {}
    rid = (data.get("rid") or "").strip()
    reply_text = (data.get("replyText") or "").strip()
    force = bool(data.get("force", False))

    if not rid:
        return jsonify({"error": "missing rid"}), 400
    if not reply_text:
        return jsonify({"error": "missing replyText"}), 400
    if len(reply_text) > 3500:
        return jsonify({"error": "reply too long"}), 400

    payload = prefill_get(rid)
    if not payload:
        return jsonify({"error": "rid not found"}), 404

    account_id = (payload.get("accountId") or "").strip()
    location_id = (payload.get("locationId") or "").strip()
    review_id = (payload.get("reviewId") or "").strip()

    if not (account_id and location_id and review_id):
        return jsonify({"error": "missing GBP ids in prefill payload"}), 400

    # Schutz: nicht überschreiben, wenn bereits geantwortet
    try:
        r = gbp_get_review(account_id, location_id, review_id)
        existing = (r.get("reviewReply") or {}).get("comment") if isinstance(r.get("reviewReply"), dict) else None
        if existing and not force:
            return jsonify({
                "error": "already_replied",
                "message": "Für diese Rezension existiert bereits eine Antwort. force=true zum Überschreiben.",
                "existingReply": existing
            }), 409
    except Exception as e:
        return jsonify({"error": "fetch_review_failed", "detail": str(e)}), 502

    try:
        out = gbp_update_reply(account_id, location_id, review_id, reply_text)
        return jsonify({
            "ok": True,
            "accountId": account_id,
            "locationId": location_id,
            "reviewId": review_id,
            "reply": out
        })
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

    if rid:
        payload = prefill_get(rid)
        if payload:
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
        publish_enabled=_gbp_enabled(),
        basic_auth_required=_basic_auth_required(),
    )


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form
    rid = (form.get("rid") or "").strip()
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

        replies.append({"review": review_text, "reply": public_answer, "insights": insights})

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
                data_bytes = json.dumps(payload).encode("utf-8")
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

    # RID NICHT löschen – wir brauchen es ggf. fürs "Publish to Google"
    return render_template(
        "index.html",
        values=values,
        reviews=review_blocks,
        replies=replies,
        rid=rid,
        publish_enabled=_gbp_enabled(),
        basic_auth_required=_basic_auth_required(),
    )


if DATABASE_URL:
    try:
        prefill_init()
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
