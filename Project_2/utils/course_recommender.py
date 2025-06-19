# Course recommendations
def recommend_courses(domain):
    suggestions = {
        "technology": ["B.Tech CSE", "Data Science Certification", "Ethical Hacking"],
        "arts": ["Graphic Design", "Fine Arts", "Animation & VFX"],
        "commerce": ["CA", "MBA Finance", "Stock Market Analysis"],
        "medical": ["MBBS", "Pharmacy", "Nursing"]
    }
    return suggestions.get(domain.lower(), ["Explore general aptitude and interest."])
