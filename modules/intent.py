import re

class IntentDetector:
    def __init__(self, llm_engine=None):
        """
        llm_engine: An instance of the LLM wrapper to use for complex intent detection.
        """
        self.llm_engine = llm_engine
        
        # Strong rule-based intents (Immediate triggers)
        self.rules = {
            "Refund Request": [r"refund", r"money back", r"return"],
            "Technical Support": [r"broken", r"not working", r"error", r"issue", r"fix"],
            "Complaint": [r"bad service", r"angry", r"terrible", r"worst"]
        }

    def detect(self, text):
        """
        Detects the intent of the conversation.
        Priority: 1. LLM (Accurate/Specific) 2. Rules (Fast Fallback)
        """
        # 1. LLM-based check (Primary for V2)
        if self.llm_engine:
            try:
                intent = self.llm_engine.get_intent(text)
                if intent and "Error" not in intent:
                    return intent
            except:
                pass

        # 2. Rule-based check (Fallback)
        for intent, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    return intent
                
        return "Uncategorized Interaction"

if __name__ == "__main__":
    id = IntentDetector()
    print(f"Intent 1: {id.detect('I want my money back for this broken item')}")
    print(f"Intent 2: {id.detect('How much does the premium plan cost?')}")
