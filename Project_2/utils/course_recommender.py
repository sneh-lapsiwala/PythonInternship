def recommend_courses(domain):
    courses_dict = {
        "technology": {
            "Top University Programs": [
                "B.Tech CSE – IIT Bombay",
                "B.Tech in AI & Data Science – VIT Vellore",
                "B.Sc Computer Science – Delhi University"
            ],
            "Online Courses": [
                "CS50 – Intro to Computer Science (Harvard)",
                "Full Stack Web Development – Udemy",
                "Google IT Support – Coursera"
            ]
        },
        "commerce": {
            "Top University Programs": [
                "B.Com – SRCC, Delhi University",
                "BBA – NMIMS Mumbai",
                "Chartered Accountancy – ICAI"
            ],
            "Online Courses": [
                "Financial Markets – Yale",
                "Accounting Fundamentals – Coursera",
                "Investment Banking – Udemy"
            ]
        },
        "arts": {
            "Top University Programs": [
                "BA in Fine Arts – MSU Baroda",
                "Bachelor of Design – NID Ahmedabad",
                "Performing Arts – BHU Varanasi"
            ],
            "Online Courses": [
                "Graphic Design Basics – Canva",
                "Digital Art Masterclass – Udemy",
                "Art History – Khan Academy"
            ]
        },
        "medical": {
            "Top University Programs": [
                "MBBS – AIIMS Delhi",
                "BDS – Maulana Azad Dental College",
                "B.Sc Nursing – CMC Vellore"
            ],
            "Online Courses": [
                "Intro to Biology – edX",
                "Anatomy Specialization – Coursera",
                "Clinical Research – Stanford Online"
            ]
        }
    }

    return courses_dict.get(domain.lower(), {
        "Top University Programs": ["Explore general aptitude or consult a counselor."],
        "Online Courses": ["Try platforms like Coursera, edX, or Khan Academy."]
    })
