import time

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = None
        self.label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive",
            "NEGATIVE": "Negative",
            "NEUTRAL": "Neutral",
            "POSITIVE": "Positive"
        }
        try:
            from transformers import pipeline
            self.analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment"
            )
        except Exception as e:
            print(f"Warning: Could not initialize HF pipeline ({e}), using lexical analyzer fallback.")
            self.analyzer = None

    def analyze(self, text):
        """
        Analyzes the sentiment of the given text.
        Returns a dictionary with score, label, and latency.
        """
        start_time = time.time()
        
        if self.analyzer:
            try:
                results = self.analyzer(text[:2000])
                latency = (time.time() - start_time) * 1000
                res = results[0]
                label = self.label_map.get(res['label'], res['label'])
                return {
                    "score": round(float(res['score']), 3),
                    "label": label,
                    "latency_ms": round(latency, 2)
                }
            except Exception:
                pass
                
        # Heuristic / Lexical fallback
        text_lower = text.lower()
        pos_words = ["great", "excellent", "happy", "love", "thanks", "perfect", "good", "helpful", "interested", "appreciate", "fantastic"]
        neg_words = ["bad", "terrible", "angry", "broken", "issue", "problem", "expensive", "error", "cancel", "refund", "worst", "unhappy", "frustrated"]
        
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        
        latency = (time.time() - start_time) * 1000
        
        if pos_count > neg_count + 1:
            label = "Positive"
            score = 0.88
        elif neg_count > pos_count:
            label = "Negative"
            score = 0.82
        else:
            label = "Neutral"
            score = 0.65
            
        return {
            "score": score,
            "label": label,
            "latency_ms": round(latency, 2)
        }

if __name__ == "__main__":
    sa = SentimentAnalyzer()
    print(sa.analyze("I am extremely happy with the service!"))
    print(sa.analyze("This is a complete disaster, I want a refund."))
