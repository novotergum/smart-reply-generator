import re
import xml.etree.ElementTree as ET
from typing import Mapping, Any


def build_prompt(user_input: Mapping[str, Any]) -> str:
    """
    Liest prompt.xml ein, evaluiert die Bedingungen und baut daraus den endgültigen Prompt
    für das Sprachmodell.
    """
    tree = ET.parse("prompt.xml")
    root = tree.getroot()

    prompt_lines: list[str] = []

    # Alle <line>-Elemente prüfen
    for line in root.findall("line"):
        condition = line.attrib.get("condition")
        if evaluate_condition(condition, user_input):
            text = (line.text or "").strip()
            if text:
                text = substitute_placeholders(text, user_input)
                prompt_lines.append(text)

    # Bewertungstext anhängen
    review_text = user_input.get("review", "")
    prompt_lines.append(
        "Hier ist die Bewertung, auf die du bitte antwortest:\n\n" + review_text
    )

    # Alles zu einem Prompt zusammenfügen
    return "\n\n".join(prompt_lines)


def substitute_placeholders(text: str, user_input: Mapping[str, Any]) -> str:
    """
    Ersetzt Platzhalter der Form {{ feldname }} durch den jeweiligen Wert
    aus user_input, falls vorhanden.
    """

    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        return str(user_input.get(key, ""))

    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", repl, text)


def evaluate_condition(condition: str | None, user_input: Mapping[str, Any]) -> bool:
    """
    Unterstützte Syntax:
      - kein condition-Attribut    -> Zeile wird immer verwendet
      - "isset:feldname"           -> Zeile wird verwendet, wenn Feld gesetzt und nicht leer ist
      - "if:feldname=wert"         -> Zeile wird verwendet, wenn user_input[feldname] == "wert"
    """
    if not condition:
        return True

    # Bedingung: isset:<field>
    if condition.startswith("isset:"):
        key = condition.split(":", 1)[1]
        value = user_input.get(key)
        return bool(value)

    # Bedingung: if:<field>=<value>
    if condition.startswith("if:"):
        expr = condition[3:]
        if "=" in expr:
            key, value = expr.split("=", 1)
            return user_input.get(key) == value

    # Standard: Bedingung nicht erkannt -> Zeile nicht verwenden
    return False
