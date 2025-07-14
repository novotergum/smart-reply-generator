import xml.etree.ElementTree as ET

def build_prompt(user_input):
    # XML-Datei einlesen
    tree = ET.parse("prompt.xml")
    root = tree.getroot()
    prompt_lines = []

    # Alle <line>-Elemente prüfen
    for line in root.findall("line"):
        condition = line.attrib.get("condition")
        if evaluate_condition(condition, user_input):
            prompt_lines.append(line.text)

    # Bewertungstext anhängen
    prompt_lines.append("Here is the review: " + user_input.get("review", ""))
    return "\n".join(prompt_lines)

def evaluate_condition(condition, user_input):
    if condition is None:
        return True

    # Bedingung: isset:<field>
    if condition.startswith("isset:"):
        key = condition.split(":")[1]
        return key in user_input and user_input[key]

    # Bedingung: if:<field>=<value>
    if condition.startswith("if:"):
        parts = condition[3:].split("=")
        if len(parts) == 2:
            key, value = parts
            return user_input.get(key) == value

    return False
