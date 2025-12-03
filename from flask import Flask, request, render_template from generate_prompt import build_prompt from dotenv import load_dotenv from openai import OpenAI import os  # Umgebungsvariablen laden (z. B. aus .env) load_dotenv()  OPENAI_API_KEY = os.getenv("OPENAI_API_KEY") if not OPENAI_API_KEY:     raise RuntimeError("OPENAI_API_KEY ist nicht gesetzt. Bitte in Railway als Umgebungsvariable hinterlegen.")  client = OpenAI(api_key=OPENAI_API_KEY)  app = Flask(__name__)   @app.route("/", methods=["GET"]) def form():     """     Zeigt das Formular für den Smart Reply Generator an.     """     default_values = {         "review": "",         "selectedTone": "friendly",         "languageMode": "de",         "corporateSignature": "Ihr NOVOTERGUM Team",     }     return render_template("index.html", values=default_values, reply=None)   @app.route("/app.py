from flask import Flask, request, render_template
from generate_prompt import build_prompt
from dotenv import load_dotenv
from openai import OpenAI
import os

# Umgebungsvariablen laden (z. B. aus .env)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt. Bitte in Railway als Umgebungsvariable hinterlegen.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def form():
    """
    Zeigt das Formular für den Smart Reply Generator an.
    """
    default_values = {
        "review": "",
        "selectedTone": "friendly",
        "languageMode": "de",
        "corporateSignature": "Ihr NOVOTERGUM Team",
    }
    return render_template("index.html", values=default_values, reply=None)


@app.route("/generate", methods=["POST"])
def generate():
    """
    Nimmt die Formulareingaben entgegen, baut den Prompt und ruft das OpenAI-Modell auf.
    """
    # request.form ist ein ImmutableMultiDict -> in ein normales dict umwandeln
    user_input = request.form.to_dict()

    prompt = build_prompt(user_input)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # bei Bedarf auf ein aktuelleres Modell anpassen
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
        # Einfache Fehlerrückgabe im Frontend
        reply = f"Es ist ein Fehler bei der Generierung der Antwort aufgetreten: {e}"
        status_code = 500

    return render_template("index.html", reply=reply, values=user_input), status_code


if __name__ == "__main__":
    # Für lokale Entwicklung. In Railway wird ein WSGI-Server (z. B. gunicorn) verwendet.
    app.run(host="0.0.0.0", port=5000, debug=True)
