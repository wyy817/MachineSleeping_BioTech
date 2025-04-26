import random


def mock_llm(symptoms):
    """
    Simulate the LLM's AI consultation based on symptoms provided by the user.
    This includes more complex rule-matching and simulates multi-turn dialogue.
    Eventually, this will be replaced with ChatGLM2-6B or another large-scale LLM.
    """
    # Rule-based symptom matching
    if "fever" in symptoms or "cough" in symptoms:
        return random.choice([
            "It could be a cold or flu, please drink plenty of water and monitor your temperature. If it worsens, seek medical attention.",
            "I suggest you rest, drink warm water, and observe if you have other symptoms like sore throat or headache."
        ])
    elif "headache" in symptoms:
        return random.choice([
            "It might be a migraine or tension-type headache. I suggest resting and avoiding trigger foods.",
            "The headache could be caused by stress or fatigue, I recommend relaxation and rest."
        ])
    elif "coughing up phlegm" in symptoms:
        return random.choice([
            "Coughing up phlegm may indicate a respiratory infection, it's best to visit a doctor for further checks.",
            "If the phlegm is green or yellow, it could be a bacterial infection. Seek medical attention."
        ])
    elif "chest pain" in symptoms:
        return "Chest pain is a serious symptom and could involve various conditions. It’s important to seek medical attention immediately, including checking for heart-related issues."
    elif "dizziness" in symptoms:
        return random.choice([
            "It could be due to low blood sugar, anemia, or fatigue. I recommend resting and consuming high-sugar food.",
            "Dizziness might be caused by an inner ear issue, avoid sudden standing and avoid vigorous physical activity."
        ])
    else:
        return "Please provide more detailed symptoms for a more accurate diagnosis."


def further_questions(symptom_input):
    """
    Simulate a multi-turn conversation by generating follow-up questions based on user input.
    """
    questions = [
        "How long have you been experiencing these symptoms?",
        "Do you have other symptoms like sore throat, muscle aches, etc.?",
        "What is your temperature? Is it accompanied by a headache or nausea?",
        "Do you have any allergies, or a history of chronic diseases?"
    ]
    return random.choice(questions)
