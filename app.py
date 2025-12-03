import os

from flask import Flask, request, render_template
from dotenv import load_dotenv
from openai import OpenAI

from generate_prompt import build_prompt

# Umgebungsvariablen laden (lokal über .env, auf Railway über Variables)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY ist nicht gesetzt. Bitte als Umgebungsvariable hinterlegen "
        "(lokal in .env, auf Railway unter Variables)."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Flask-App initialisieren, Templates liegen im Ordner "templates"
app = Flask(__name__, template_folder="templates")


def derive_rating_category(user_input: dict) -> None:
    """
    Leitet aus ratingStars automatisch eine ratingCategory ab,
    falls noch keine gesetzt ist.

    - 4–5 Sterne -> positiv
    - 3 Sterne   -> neutral
    - 1–2 Sterne -> kritisch
    """
    if user_input.get("ratingCategory"):
        # Falls du später manuell ein Label setzt, bleibt das bestehen.
        return

    stars_str = (user_input.get("ratingStars") or "").strip()
    if not stars_str.isdigit():
        return

    stars = int(stars_str)
    if stars >= 4:
        user_input["ratingCategory"] = "positiv"
    elif stars == 3:
        user_input["ratingCategory"] = "neutral"
    elif 1 <= stars <= 2:
        user_input["ratingCategory"] = "kritisch"


@app.route("/health", methods=["GET"])
def health():
    """
    Einfache Health-Check-Route für Monitoring.
    """
    return "ok", 200


@app.route("/", methods=["GET"])
def form():
    """
    Zeigt das Formular für den Smart Reply Generator an.
    """
    default_values = {
        "review": "",
        "selectedTone": "friendly",
        "languageMode": "de",  # standardmäßig Deutsch
        "corporateSignature": "Ihr NOVOTERGUM Team",

        "ratingStars": "",
        "ratingCategory": "",
        "reviewTag": "",
        "anrede": "",
        "kontaktMail": "",
    }
    return render_template("index.html", values=default_values, reply=None)


@app.route("/generate", methods=["POST"])
def generate():
    """
    Nimmt die Formulareingaben entgegen, baut den Prompt und ruft das OpenAI-Modell auf.
    """
    user_input = request.form.to_dict()

    # Sprache hart auf Deutsch setzen, falls nichts kommt
    user_input.setdefault("languageMode", "de")

    # Rating-Kategorie aus den Sternen ableiten (falls möglich)
    derive_rating_category(user_input)

    prompt = build_prompt(user_input)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        reply = response.choices[0].message.content
        status_code = 200
    except Exception as e:
        reply = (
            "Es ist ein Fehler bei der Generierung der Antwort aufgetreten: "
            f"{e}"
        )
        status_code = 500

    return render_template("index.html", values=user_input, reply=reply), status_code


if __name__ == "__main__":
    # Lokaler Start. Auf Railway übernimmt ein WSGI-Server (z. B. gunicorn).
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
