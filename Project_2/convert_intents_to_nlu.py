# convert_intents_to_nlu.py

import json
from pathlib import Path
import yaml

# Load intents.json
json_path = Path("data/intents.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare NLU structure
nlu_data = []

for intent in data["intents"]:
    tag = intent["tag"]
    for pattern in intent.get("patterns", []):
        nlu_data.append({
            "intent": tag,
            "examples": f"- {pattern}"
        })

# Combine examples under the same intent
intent_dict = {}
for entry in nlu_data:
    intent = entry["intent"]
    example = entry["examples"]
    intent_dict.setdefault(intent, []).append(example)

rasa_nlu = {
    "version": "3.1",
    "nlu": []
}

for intent, examples in intent_dict.items():
    rasa_nlu["nlu"].append({
        "intent": intent,
        "examples": "\n".join(examples)
    })

# Save to nlu.yml
output_path = Path("rasa_project/data/nlu.yml")
with open(output_path, "w", encoding="utf-8") as f:
    yaml.dump(rasa_nlu, f, sort_keys=False, allow_unicode=True)

print("✅ nlu.yml generated from intents.json!")
