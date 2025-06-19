# MBTI/RIASEC logic
def map_mbti_to_domain(mbti_type):
    if mbti_type.startswith("I"):
        return "technology"
    elif mbti_type.startswith("E"):
        return "commerce"
    elif "F" in mbti_type:
        return "arts"
    else:
        return "medical"
