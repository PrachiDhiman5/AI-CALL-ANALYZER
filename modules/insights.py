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
        Runs the full multi-stage intelligence pipeline on a call transcript.
        """
        self.intent_model.llm_engine = self.llm_engine
        
        metrics = {}
        start_all = time.time()
        
        # 1. Sentiment Analysis
        sentiment = self.sentiment_model.analyze(transcript)
        metrics['sentiment_latency'] = sentiment.get('latency_ms', 0)
        
        # 2. Keyword Extraction
        keywords = self.llm_engine.extract_business_keywords(transcript)
        
        # 3. Intent Detection
        intent = self.intent_model.detect(transcript)
        
        # 4. RAG: Retrieve Similar Past Conversations
        emb_start = time.time()
        try:
            query_vec, _ = self.embedder.get_embeddings([transcript])
            similar_calls = self.vector_db.search(query_vec[0], k=2)
        except Exception:
            similar_calls = []
        metrics['rag_latency'] = round((time.time() - emb_start) * 1000, 2)
        
        rag_context = "\n---\n".join(similar_calls) if similar_calls else "No similar past cases found."
        
        # 5. LLM Insights
        llm_start = time.time()
        insights = self.llm_engine.generate_insights(
            transcript, 
            sentiment.get('label', 'Neutral'), 
            keywords, 
            rag_context
        )
        metrics['llm_latency'] = round((time.time() - llm_start) * 1000, 2)
        
        # 6. Speech Diarization & Turn Structuring (New Feature!)
        turns = self.llm_engine.diarize_and_timestamp_transcript(transcript)
        
        # 7. Action Items & Next Steps (New Feature!)
        action_items = self.llm_engine.extract_action_items(transcript)
        
        # 8. Objections & Battlecard Rebuttals (New Feature!)
        objections = self.llm_engine.extract_objections_and_rebuttals(transcript)
        
        # 9. Conversational Coaching Metrics (New Feature!)
        coaching = self.llm_engine.calculate_coaching_metrics(turns, sentiment.get('score', 0.7))
        
        # Calculate total latency
        metrics['total_latency'] = round((time.time() - start_all) * 1000, 2)
        
        # 10. Deal Health & Churn Risk Analysis
        risk_level = "🟢 Healthy Deal"
        sent_label = sentiment.get('label', '').strip().lower()
        intent_lower = intent.lower()
        
        if sent_label == "negative":
            risk_level = "🟡 At Risk"
            high_risk_terms = ["objection", "disinterest", "lost", "reproach", "rejection", "decline", "price", "barrier"]
            if any(term in intent_lower for term in high_risk_terms):
                risk_level = "🔴 High Churn / Loss Risk"
        elif sent_label == "neutral" and any(term in intent_lower for term in ["objection", "barrier"]):
            risk_level = "🟡 At Risk (Pricing Friction)"
        elif coaching.get('closing_probability', 70) >= 80:
            risk_level = "🟢 High-Conviction Opportunity"
        
        return {
            "source_type": source_type,
            "filename": filename,
            "transcript": transcript,
            "sentiment_score": sentiment.get('score', 0.75),
            "sentiment_label": sentiment.get('label', 'Positive'),
            "intent": intent,
            "risk_level": risk_level,
            "keywords": keywords,
            "insights": insights,
            "dialogue_turns": turns,
            "action_items": action_items,
            "objections": objections,
            "coaching_metrics": coaching,
            "latency_metrics": metrics
        }
