import os
import json
import time
import secrets
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
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage

# Google Business Profile OAuth (für echtes Publishing)
GBP_CLIENT_ID = os.getenv("GBP_CLIENT_ID", "").strip()
GBP_CLIENT_SECRET = os.getenv("GBP_CLIENT_SECRET", "").strip()
GBP_REFRESH_TOKEN = os.getenv("GBP_REFRESH_TOKEN", "").strip()

# Feature Flags
def env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")

ENABLE_PUBLISH = env_truthy("ENABLE_PUBLISH")
PUBLISH_UI_ENABLED = env_truthy("PUBLISH_UI_ENABLED")
PUBLISH_DRY_RUN = env_truthy("PUBLISH_DRY_RUN")

# CORS: Komma-separierte Origins, z.B.
# PUBLISH_ALLOWED_ORIGINS=https://smart-reply...railway.app,https://ticket.novotergum.de
PUBLISH_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("PUBLISH_ALLOWED_ORIGINS", "").split(",") if o.strip()
]

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def must_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env: {name}")
    return v

def _corsify(resp):
    origin = request.headers.get("Origin", "")
    # Wenn Allowlist gesetzt: nur erlauben, wenn explizit drin.
    if PUBLISH_ALLOWED_ORIGINS:
        if origin in PUBLISH_ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
    else:
        # lockerer Modus: Origin spiegeln (für interne iFrame/Dev-Setups)
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"

    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Prefill-Secret"
    return resp

def _json(data, status=200):
    resp = make_response(jsonify(data), status)
    return _corsify(resp)

def _utf8_len(s: str) -> int:
    return len((s or "").encode("utf-8"))

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
                    used_at BIGINT,
                    used_count INT DEFAULT 0,
                    generated JSONB,
                    generated_at BIGINT
                )
            """)
            # "Migrations" für bestehende Installationen:
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS published_at BIGINT")
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS publish_result JSONB")
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

            payload_raw, generated_raw, created_at, published_at, publish_result_raw = row
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            generated = json.loads(generated_raw) if isinstance(generated_raw, str) else generated_raw
            publish_result = (
                json.loads(publish_result_raw) if isinstance(publish_result_raw, str) else publish_result_raw
            )

            return {
                "payload": payload,
                "generated": generated,
                "created_at": created_at,
                "published_at": published_at,
                "publish_result": publish_result,
            }

def prefill_set_generated(rid: str, data: dict):
    if not rid:
        return
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET generated=%s,generated_at=%s WHERE rid=%s",
                (json.dumps(data, ensure_ascii=False), int(time.time()), rid),
            )
        conn.commit()

def prefill_set_published(rid: str, publish_result: dict):
    if not rid:
        return
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET published_at=%s, publish_result=%s WHERE rid=%s",
                (int(time.time()), json.dumps(publish_result, ensure_ascii=False), rid),
            )
        conn.commit()

# --------------------------------------------------------
# Reviewer-Deduplikation
# --------------------------------------------------------

def _suffix_line(name: str, date: str) -> str:
    parts = []
    if (name or "").strip():
        parts.append(name.strip())
    if (date or "").strip():
        parts.append(f"am {date.strip()}")
    if not parts:
        return ""
    return "— " + ", ".join(parts)

def _dedupe_reviewer(text: str, reviewer: str, reviewed_at: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    suffix = _suffix_line(reviewer, reviewed_at)
    if not suffix:
        return text

    normalized_suffix = suffix.replace("–", "—").replace("-", "—")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    while lines and lines[-1].replace("–", "—").replace("-", "—") == normalized_suffix:
        lines.pop()

    lines.append(normalized_suffix)
    return "\n".join(lines)

# --------------------------------------------------------
# Publishing readiness
# --------------------------------------------------------

def compute_publish_ready(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    p = payload or {}
    missing = []
    for k in ("accountId", "locationId", "reviewId"):
        v = p.get(k)
        if v is None or str(v).strip() == "":
            missing.append(k)
    return (len(missing) == 0), missing

# --------------------------------------------------------
# Google OAuth + Publish Calls
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

    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Token error {e.code}: {raw}")
    except URLError as e:
        raise RuntimeError(f"Token network error: {e}")

    j = json.loads(raw)
    if not j.get("access_token"):
        raise RuntimeError(f"No access_token in token response: {raw}")
    return j["access_token"]

def get_review(account_id: str, location_id: str, review_id: str) -> dict:
    """
    Google API: accounts.locations.reviews.get
    GET https://mybusiness.googleapis.com/v4/{name=accounts/*/locations/*/reviews/*}
    """
    access_token = get_access_token()
    name = f"accounts/{account_id}/locations/{location_id}/reviews/{review_id}"
    url = f"https://mybusiness.googleapis.com/v4/{name}"

    req = Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    try:
        with urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Get review error {e.code}: {raw}")
    except URLError as e:
        raise RuntimeError(f"Get review network error: {e}")

def publish_reply(account_id: str, location_id: str, review_id: str, reply_text: str) -> dict:
    """
    Google API: accounts.locations.reviews.updateReply
    PUT https://mybusiness.googleapis.com/v4/{name=accounts/*/locations/*/reviews/*}/reply
    """
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

    try:
        with urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {"ok": True}
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Publish error {e.code}: {raw}")
    except URLError as e:
        raise RuntimeError(f"Publish network error: {e}")

# --------------------------------------------------------
# API
# --------------------------------------------------------

@app.post("/api/prefill")
def api_prefill():
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(force=True) or {}

    review = (data.get("review") or "").strip()
    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()
    rating = str(data.get("rating") or "").strip()

    account_id = (data.get("accountId") or data.get("account_id") or "").strip()
    location_id = (data.get("locationId") or data.get("location_id") or "").strip()
    review_id = (data.get("reviewId") or data.get("review_id") or "").strip()

    review = _dedupe_reviewer(review, reviewer, reviewed_at)

    payload = {
        "review": review,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "rating": rating,

        "accountId": account_id,
        "locationId": location_id,
        "reviewId": review_id,

        "storeCode": data.get("storeCode"),
        "locationTitle": data.get("locationTitle"),

        "maps_uri": data.get("maps_uri"),
        "new_review_uri": data.get("new_review_uri"),
        "place_id": data.get("place_id"),
        "maps_place_url": data.get("maps_place_url"),
    }

    rid = prefill_insert(payload)

    ready, missing = compute_publish_ready(payload)
    app.logger.info(
        "prefill_in rid=%s ready=%s missing=%s accountId=%s locationId=%s reviewId=%s",
        rid,
        ready,
        ",".join(missing) if missing else "-",
        "Y" if payload.get("accountId") else "N",
        "Y" if payload.get("locationId") else "N",
        "Y" if payload.get("reviewId") else "N",
    )

    return jsonify({"rid": rid})

@app.route("/api/debug/prefill", methods=["GET"])
def api_debug_prefill():
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    rid = (request.args.get("rid") or "").strip()
    row = prefill_get_row(rid)
    if not row:
        return jsonify({"ok": False, "error": "rid not found"}), 404

    p = row.get("payload") or {}
    ready, missing = compute_publish_ready(p)

    return jsonify({
        "ok": True,
        "rid": rid,
        "created_at": row.get("created_at"),
        "publish_ready": ready,
        "publish_missing": missing,
        "payload_keys": sorted(list(p.keys())),
        "accountId": p.get("accountId"),
        "locationId": p.get("locationId"),
        "reviewId": p.get("reviewId"),
    })

@app.route("/api/publish", methods=["OPTIONS", "POST"])
def api_publish():
    if request.method == "OPTIONS":
        return _corsify(make_response("", 204))

    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        return _json({"ok": False, "error": "unauthorized"}, 401)

    if not ENABLE_PUBLISH:
        return _json({"ok": False, "error": "publishing disabled"}, 403)

    rid = (request.args.get("rid") or "").strip()
    body = request.get_json(silent=True) or {}

    if not rid:
        rid = (body.get("rid") or "").strip()

    if not rid:
        return _json({"ok": False, "error": "missing rid"}, 400)

    row = prefill_get_row(rid)
    if not row:
        return _json({"ok": False, "error": "rid not found"}, 404)

    created_at = int(row.get("created_at") or 0)
    if created_at and int(time.time()) - created_at > PREFILL_TTL_SECONDS:
        return _json({"ok": False, "error": "rid expired"}, 410)

    payload = row.get("payload") or {}
    publish_ready, missing = compute_publish_ready(payload)
    if not publish_ready:
        return _json({"ok": False, "error": "publish not ready", "missing": missing}, 400)

    overwrite = str(body.get("overwrite") or "").strip().lower() in ("1", "true", "yes", "on")

    # Reply-Text: bevorzugt UI (damit Edit nachträglich funktioniert)
    reply_text = (body.get("reply") or "").strip()

    # Fallback: generierte Antwort aus DB
    if not reply_text:
        generated = row.get("generated") or {}
        try:
            replies = (generated.get("replies") or [])
            if replies:
                reply_text = (replies[0].get("reply") or "").strip()
        except Exception:
            reply_text = ""

    if not reply_text:
        return _json({"ok": False, "error": "no reply text available"}, 400)

    # Guard: Reply-Limit (praktisch: früh failen)
    if _utf8_len(reply_text) > 4096:
        return _json({"ok": False, "error": "reply_too_long", "max_bytes": 4096}, 400)

    account_id = str(payload.get("accountId") or "").strip()
    location_id = str(payload.get("locationId") or "").strip()
    review_id = str(payload.get("reviewId") or "").strip()

    # Serverseitiger Konflikt-Check (Race Conditions)
    existing_reply = None
    existing_updated_at = None
    try:
        review_obj = get_review(account_id, location_id, review_id)
        rr = (review_obj.get("reviewReply") or {})
        existing_reply = (rr.get("comment") or "").strip() or None
        existing_updated_at = rr.get("updateTime") or None
    except Exception as e:
        # Wenn Check nicht möglich (z.B. fehlende OAuth-Creds), nicht hart blocken.
        app.logger.warning("publish_precheck_failed rid=%s err=%s", rid, str(e))

    if existing_reply and not overwrite:
        # Idempotenz: wenn identisch, OK zurück
        if existing_reply == reply_text:
            prefill_set_published(rid, {"dry_run": bool(PUBLISH_DRY_RUN), "already_up_to_date": True})
            return _json({"ok": True, "already_up_to_date": True, "existingReplyUpdatedAt": existing_updated_at}, 200)

        return _json({
            "ok": False,
            "error": "already_replied",
            "existingReply": existing_reply,
            "existingReplyUpdatedAt": existing_updated_at,
            "hint": "Set overwrite=true to replace the existing reply."
        }, 409)

    # Dry Run
    if PUBLISH_DRY_RUN:
        app.logger.info(
            "publish_dry_run rid=%s accountId=%s locationId=%s reviewId=%s reply_len=%s overwrite=%s",
            rid, account_id, location_id, review_id, len(reply_text), overwrite
        )
        prefill_set_published(rid, {"dry_run": True, "rid": rid, "overwrite": overwrite})
        return _json({"ok": True, "dry_run": True, "message": "Dry-Run OK (siehe Railway Logs)"}, 200)

    # Echt veröffentlichen (UpdateReply überschreibt/erstellt)
    try:
        result = publish_reply(account_id, location_id, review_id, reply_text)
        prefill_set_published(rid, {"dry_run": False, "result": result, "overwrite": overwrite})
        return _json({"ok": True, "dry_run": False, "result": result}, 200)
    except Exception as e:
        app.logger.exception("publish_error rid=%s", rid)
        return _json({"ok": False, "error": str(e)}, 500)

# --------------------------------------------------------
# Index + Generator
# --------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()
    reviews, replies = [{}], None

    publish_ready = False
    publish_missing: List[str] = []
    prefill_mode = bool(rid)

    published_at = None
    publish_result = None

    if rid:
        row = prefill_get_row(rid)
        if row and row.get("payload"):
            p = row["payload"]
            publish_ready, publish_missing = compute_publish_ready(p)

            reviews = [{
                "review": _dedupe_reviewer(p.get("review", ""), p.get("reviewer", ""), p.get("reviewed_at", "")),
                "rating": p.get("rating", ""),
                "reviewType": "",
                "salutation": "",
            }]

            if row.get("generated"):
                replies = (row["generated"] or {}).get("replies")

            published_at = row.get("published_at")
            publish_result = row.get("publish_result")

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

        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
        publish_ready=publish_ready,
        publish_missing=publish_missing,
        publish_dry_run=PUBLISH_DRY_RUN,

        published_at=published_at,
        publish_result=publish_result,
    )

def _first_non_empty_pairs(reviews: List[str], ratings: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for idx, rev in enumerate(reviews[:MAX_REVIEWS]):
        rev_txt = (rev or "").strip()
        if not rev_txt:
            continue
        rat = (ratings[idx] if idx < len(ratings) else "") or ""
        pairs.append((rev_txt, str(rat)))
    return pairs

@app.post("/generate")
def generate():
    rid = (request.form.get("rid") or "").strip()
    reviews = request.form.getlist("review")
    ratings = request.form.getlist("rating")

    values = {
        "selectedTone": request.form.get("selectedTone", "friendly"),
        "corporateSignature": request.form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
        "contactEmail": request.form.get("contactEmail", ""),
        "languageMode": request.form.get("languageMode", "de"),
    }

    pairs = _first_non_empty_pairs(reviews, ratings)

    # rid = Single-Review Modus
    if rid and pairs:
        if len(pairs) > 1:
            app.logger.warning("generate rid=%s received %s reviews; forcing single-review mode", rid, len(pairs))
        pairs = [pairs[0]]

    replies_out = []
    for (rev, rating) in pairs:
        prompt = build_prompt({
            "review": rev,
            "rating": rating,
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
        replies_out.append({"review": rev, "reply": public, "insights": insights})

    if rid:
        prefill_set_generated(rid, {"replies": replies_out})
        return redirect(url_for("index", rid=rid))

    return render_template(
        "index.html",
        values=values,
        reviews=[{"review": r} for r in reviews],
        replies=replies_out,
        rid=rid,
        prefill_mode=False,

        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,
        publish_ready=False,
        publish_missing=["accountId", "locationId", "reviewId"],
        publish_dry_run=PUBLISH_DRY_RUN,

        published_at=None,
        publish_result=None,
    )

# --------------------------------------------------------
# Start
# --------------------------------------------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
