from transformers import pipeline
import time

class SentimentAnalyzer:
    def __init__(self):
        print("Initializing Advanced Sentiment Analysis model...")
        # Use a 3-class model (Positive, Neutral, Negative) for better nuance
        self.analyzer = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        # Mapping labels for twitter-roberta (LABEL_0: Negative, LABEL_1: Neutral, LABEL_2: Positive)
        self.label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }

    def analyze(self, text):
        """
        Analyzes the sentiment of the given text.
        Returns a dictionary with score, label, and latency.
        """
        start_time = time.time()
        
        # Split text into chunks if it's too long (Transformers limitation)
        # For simplicity, we'll take the first 512 tokens
        results = self.analyzer(text[:2000]) 
        
        latency = (time.time() - start_time) * 1000 # in ms
        
        res = results[0]
        label = self.label_map.get(res['label'], res['label'])
        
        return {
            "score": res['score'],
            "label": label,
            "latency_ms": round(latency, 2)
        }

if __name__ == "__main__":
    sa = SentimentAnalyzer()
    print(sa.analyze("I am extremely happy with the service!"))
    print(sa.analyze("This is a complete disaster, I want a refund."))
