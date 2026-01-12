import xml.etree.ElementTree as ET
import json
import re
from typing import Dict, Any, Optional, Tuple

# --------------------------------------------------------
# Prompt-Aufbau aus prompt.xml + Form-Daten
# --------------------------------------------------------


def build_prompt(user_input: Dict[str, Any]) -> str:
    """
    Baut den vollständigen Prompt für genau EINE Bewertung.
    Erwartet ein dict mit Feldern wie:
      - review
      - rating
      - reviewType
      - salutation
      - corporateSignature
      - contactEmail
      - selectedTone
      - languageMode
    """
    # XML-Datei einlesen
    tree = ET.parse("prompt.xml")
    root = tree.getroot()
    prompt_lines = []

    # Alle <line>-Elemente prüfen
    for node in root.findall("line"):
        condition = node.attrib.get("condition")
        if evaluate_condition(condition, user_input):
            text = (node.text or "").strip()
            if text:
                prompt_lines.append(text)

    # Form-Werte auslesen (Fallbacks bewusst defensiv)
    review_text = (user_input.get("review") or "").strip()
    rating = (user_input.get("rating") or "").strip()
    review_type = (user_input.get("reviewType") or "").strip()
    salutation = (user_input.get("salutation") or "").strip()
    corporate_signature = (user_input.get("corporateSignature") or "").strip()
    contact_email = (user_input.get("contactEmail") or "").strip()
    selected_tone = (user_input.get("selectedTone") or "").strip()
    language_mode = (user_input.get("languageMode") or "de").strip()

    # Strukturierter Kontext zur Bewertung
    context_lines = [
        "Hier sind die konkreten Daten dieser einzelnen Bewertung:",
        "",
        f"- Bewertungstext: {review_text or '(leer)'}",
        f"- Sternebewertung (1–5, falls gesetzt): {rating or '(nicht gesetzt)'}",
        f"- Art der Bewertung (reviewType): {review_type or '(nicht gesetzt)'}",
        f"- Anrede (salutation): {salutation or '(nicht gesetzt)'}",
        f"- Signatur (corporateSignature): {corporate_signature or '(nicht gesetzt)'}",
        f"- Kontakt-E-Mail (contactEmail): {contact_email or '(nicht gesetzt)'}",
        f"- Ausgewählter Tonfall (selectedTone): {selected_tone or '(nicht gesetzt)'}",
        f"- Sprache (languageMode): {language_mode}",
        "",
        (
            "Erzeuge jetzt die Ausgabe exakt in dem beschriebenen Format mit:\n"
            "1) ÖFFENTLICHE ANTWORT:\n"
            "   [öffentlicher Antworttext]\n"
            "\n"
            "2) INTERNE INSIGHTS (NICHT VERÖFFENTLICHEN):\n"
            "   ```json\n"
            "   { ... gültiges JSON gemäß Schema ... }\n"
            "   ```"
        ),
    ]

    prompt = "\n".join(prompt_lines + [""] + context_lines).strip()
    return prompt


def evaluate_condition(condition: Optional[str], user_input: Dict[str, Any]) -> bool:
    """
    Unterstützte Condition-Syntax in prompt.xml:
      - kein / None      -> Zeile wird immer verwendet
      - "isset:feld"     -> feld ist vorhanden und nicht leer
      - "if:feld=wert"   -> feld hat exakt den String wert
    """
    if condition is None or condition.strip() == "":
        return True

    condition = condition.strip()

    # Bedingung: isset:<field>
    if condition.startswith("isset:"):
        key = condition.split(":", 1)[1]
        val = user_input.get(key)
        return val is not None and str(val).strip() != ""

    # Bedingung: if:<field>=<value>
    if condition.startswith("if:"):
        expr = condition[3:]
        if "=" not in expr:
            return False
        key, expected = expr.split("=", 1)
        key = key.strip()
        expected = expected.strip()
        actual = str(user_input.get(key, "")).strip()
        return actual == expected

    # Unbekannte Condition -> sicherheitshalber False
    return False


# --------------------------------------------------------
# Helper: Öffentliche Antwort + Interne Insights aus dem
# Modell-Output extrahieren
# --------------------------------------------------------

INSIGHTS_HEADER = "INTERNE INSIGHTS (NICHT VERÖFFENTLICHEN):"


def split_public_and_insights(raw: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Zerlegt die Modellantwort in:
      - public_text: öffentliche Antwort (String)
      - insights: dict mit den internen Insights (oder None)

    Erwartetes Modell-Format:

    ÖFFENTLICHE ANTWORT:
    [öffentlicher Antworttext ...]

    INTERNE INSIGHTS (NICHT VERÖFFENTLICHEN):
    ```json
    {
      ...
    }
    ```
    """
    if not raw:
        return "", None

    # Am Insights-Header splitten
    parts = raw.split(INSIGHTS_HEADER, 1)
    if len(parts) == 1:
        # Kein Insights-Teil gefunden -> alles als öffentliche Antwort
        public_only = raw.strip()
        public_only = _strip_public_header(public_only)
        return public_only, None

    public_part = parts[0].strip()
    insights_raw = parts[1]

    # "ÖFFENTLICHE ANTWORT:" im Public-Part entfernen
    public_part = _strip_public_header(public_part)

    # JSON aus ```json ... ``` herausziehen
    json_match = re.search(r"```json\s*(\{.*\})\s*```", insights_raw, flags=re.DOTALL | re.IGNORECASE)
    if not json_match:
        # Fallback: irgendeinen {}-Block greifen
        brace_match = re.search(r"(\{.*\})", insights_raw, flags=re.DOTALL)
        if not brace_match:
            return public_part.strip(), None
        json_str = brace_match.group(1)
    else:
        json_str = json_match.group(1)

    insights = _safe_parse_json(json_str)
    return public_part.strip(), insights


def _strip_public_header(text: str) -> str:
    """
    Entfernt eine eventuell vorangestellte Zeile "ÖFFENTLICHE ANTWORT:" (case-insensitive).
    """
    return re.sub(
        r"^\s*ÖFFENTLICHE ANTWORT:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _safe_parse_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Robustes JSON-Parsing mit minimaler Bereinigung.
    Gibt bei Fehlern None zurück, statt zu crashen.
    """
    if not json_str:
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = json_str.replace("\r", " ").replace("\n", " ").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return None
