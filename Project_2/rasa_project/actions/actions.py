# EDIT: rasa_project/actions.py
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# You'll need to install requests for making API calls to Rasa server
# if you were to connect Streamlit. Here, the Streamlit app will call Rasa,
# so this actions.py doesn't need to call back to Streamlit directly.
# However, if you want Rasa to trigger actions *in* Streamlit, you'd
# set up a custom connector or a webhook. For simplicity, we'll keep
# the Streamlit app as the orchestrator.

class ActionRecommendCareer(Action):
    def name(self) -> Text:
        return "action_recommend_career"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # This action will prompt the user to provide more info, likely a resume.
        # The actual resume parsing and recommendation logic will live in the Streamlit app.
        dispatcher.utter_template("utter_generic_recommendation", tracker)
        return []

class ActionProvideCareerAdvice(Action):
    def name(self) -> Text:
        return "action_provide_career_advice"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        latest_intent = tracker.latest_message.get("intent", {}).get("name")

        if latest_intent == "technology":
            dispatcher.utter_template("utter_technology_advice", tracker)
        elif latest_intent == "arts":
            dispatcher.utter_template("utter_arts_advice", tracker)
        elif latest_intent == "commerce":
            dispatcher.utter_template("utter_commerce_advice", tracker)
        elif latest_intent == "medical":
            dispatcher.utter_template("utter_medical_advice", tracker)
        else: # For general career_advice intent or fallback
            dispatcher.utter_template("utter_default_career_advice", tracker)

        return []