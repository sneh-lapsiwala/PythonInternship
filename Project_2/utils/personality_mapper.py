# MBTI/RIASEC logic
def map_mbti_to_domain(mbti_type):
    mbti_type = mbti_type.upper()
    if "F" in mbti_type:
        return "arts"
    elif mbti_type.startswith("I"):
        return "technology"
    elif mbti_type.startswith("E"):
        return "commerce"
    else:
        return "medical"

# utils/personality_mapper.py

def map_personality(mbti_code):
    mbti_traits = {
        "I": "Introverted",
        "E": "Extraverted",
        "S": "Sensing",
        "N": "Intuitive",
        "T": "Thinking",
        "F": "Feeling",
        "J": "Judging",
        "P": "Perceiving"
    }

    description = []
    for char in mbti_code.upper():
        trait = mbti_traits.get(char)
        if trait:
            description.append(trait)
        else:
            description.append("Unknown")

    return description
