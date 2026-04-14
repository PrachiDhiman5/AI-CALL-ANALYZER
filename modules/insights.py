import time

class InsightsEngine:
    def __init__(self, sentiment_model, intent_model, keyword_extractor, embedder, vector_db, llm_engine):
        self.sentiment_model = sentiment_model
        self.intent_model = intent_model
        self.keyword_extractor = keyword_extractor
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm_engine = llm_engine

    def run_full_analysis(self, transcript, filename="N/A", source_type="Text"):
        """
        Runs the entire pipeline on a single transcript.
        """
        # Ensure Intent model uses the same dynamic LLM engine
        self.intent_model.llm_engine = self.llm_engine
        
        metrics = {}
        start_all = time.time()
        
        # 1. Sentiment Analysis
        sentiment = self.sentiment_model.analyze(transcript)
        metrics['sentiment_latency'] = sentiment['latency_ms']
        
        # 2. Keyword Extraction (Using LLM for higher quality)
        keywords = self.llm_engine.extract_business_keywords(transcript)
        
        # 3. Intent Detection
        # Pass llm_engine if rule-based fails (handled inside intent.py)
        intent = self.intent_model.detect(transcript)
        
        # 4. RAG: Retrieve Similar Past Conversations
        emb_start = time.time()
        query_vec, _ = self.embedder.get_embeddings([transcript])
        similar_calls = self.vector_db.search(query_vec[0], k=2)
        metrics['rag_latency'] = round((time.time() - emb_start) * 1000, 2)
        
        rag_context = "\n---\n".join(similar_calls) if similar_calls else "No similar past cases found."
        
        # 5. LLM Insights
        llm_start = time.time()
        insights = self.llm_engine.generate_insights(
            transcript, 
            sentiment['label'], 
            keywords, 
            rag_context
        )
        metrics['llm_latency'] = round((time.time() - llm_start) * 1000, 2)
        
        # Calculate total latency
        metrics['total_latency'] = round((time.time() - start_all) * 1000, 2)
        
        # 6. Sales Health Analysis (Hardened Logic)
        risk_level = "🟢 Healthy"
        sent_label = sentiment['label'].strip().lower()
        intent_lower = intent.lower()
        
        if sent_label == "negative":
            risk_level = "🟡 At Risk"
            # Flag high risk if it's negative AND involves an objection, rejection, or cost
            high_risk_terms = ["objection", "disinterest", "lost", "reproach", "rejection", "decline", "price"]
            if any(term in intent_lower for term in high_risk_terms):
                risk_level = "🔴 High Retention Risk"
        elif sent_label == "neutral" and "objection" in intent_lower:
            risk_level = "🟡 At Risk (Neutral Objection)"
        
        return {
            "source_type": source_type,
            "filename": filename,
            "transcript": transcript,
            "sentiment_score": sentiment['score'],
            "sentiment_label": sentiment['label'],
            "intent": intent,
            "risk_level": risk_level,
            "keywords": keywords,
            "insights": insights,
            "latency_metrics": metrics
        }
