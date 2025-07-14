from flask import Flask, request, render_template
from generate_prompt import build_prompt
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def form():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.form
    prompt = build_prompt(user_input)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content
    return render_template("index.html", reply=reply, values=user_input)

if __name__ == "__main__":
    app.run(debug=True)
