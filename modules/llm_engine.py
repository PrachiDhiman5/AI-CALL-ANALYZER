import os
import json
import time
from groq import Groq

CANDIDATE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "groq/compound"
]

class LLMEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
        self.models = CANDIDATE_MODELS

    def get_completion(self, system_prompt, user_prompt, temperature=0.3):
        if not self.client:
            return "Error: Groq API Key missing."
            
        last_err = None
        for model_name in self.models:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model_name,
                    temperature=temperature,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                last_err = str(e)
                continue
        return f"LLM Error: {last_err}"

    def get_intent(self, transcript):
        system_prompt = """
        Analyze the call transcript and identify the SPECIFIC primary intent of the customer.
        NEVER return generic categories like 'Sales Inquiry' or 'Inquiry'.
        Be highly specific and differentiate between interest and rejection.
        Use detailed business categories such as: 
        - Pricing Objection / Cost Barrier
        - Disinterest / Lead Lost
        - Competitor Comparison & Evaluation
        - Technical Troubleshooting & Bug Report
        - Product Feature & Capability Inquiry
        - Qualified High-Intent Sales Prospect
        - Churn Risk / Subscription Cancellation
        - Refund & Billing Dispute
        
        If the customer is rejecting or hesitating due to price/budget, ALWAYS include the word 'Objection' or 'Barrier'.
        Return ONLY the 2-4 word category name. No quotes or punctuation.
        """
        res = self.get_completion(system_prompt, f"Transcript: {transcript}")
        return res.strip().replace(".", "").replace('"', '')

    def extract_business_keywords(self, transcript):
        """
        Extracts meaningful business keywords using LLM intelligence.
        """
        system_prompt = "Extract 5-8 highly relevant business keywords or phrases from this call transcript. Focus on customer pain points, product names, objections, and buying criteria. Return them as a comma-separated list with no numbering."
        res = self.get_completion(system_prompt, f"Transcript: {transcript}")
        keywords = [k.strip().replace("-", "").strip() for k in res.split(',') if k.strip()]
        return keywords[:8]

    def generate_insights(self, transcript, sentiment, keywords, rag_context=""):
        system_prompt = """
        You are an Elite Enterprise Sales & Customer Intelligence Analyst.
        Generate structured, high-value executive insights from this call transcript.
        
        Provide your response in Markdown with these exact sections:
        - **Executive Summary**: 2 concise sentences summarizing the primary dialogue objective and outcome.
        - **Key Pain Points & Objections**: 2-3 specific bullet points detailing the customer's friction or queries.
        - **Sales Coaching & Repertoire**: What did the representative do well, and where could they improve?
        - **Next Steps & Deal Velocity**: What concrete action should be executed next to advance or resolve this case?
        """
        
        user_prompt = f"""
        Transcript: {transcript}
        Sentiment: {sentiment}
        Key Themes: {', '.join(keywords)}
        Historical Reference Context: {rag_context}
        """
        
        return self.get_completion(system_prompt, user_prompt)

    def extract_action_items(self, transcript):
        """
        Extracts action items, follow-ups, and commitments from the call.
        """
        system_prompt = """
        Extract concrete action items, commitments, and next steps agreed upon during this call.
        Return ONLY a JSON array of objects with keys: "assignee" (e.g. "Sales Manager" or "Customer"), "task" (short description), "priority" ("High", "Medium", "Low").
        Example:
        [
          {"assignee": "Sales Manager", "task": "Send custom enterprise pricing tiers and ROI calculator", "priority": "High"},
          {"assignee": "Customer", "task": "Review security whitepaper with CTO before Thursday", "priority": "Medium"}
        ]
        Return ONLY raw valid JSON. No markdown code blocks.
        """
        res = self.get_completion(system_prompt, f"Transcript: {transcript}")
        try:
            cleaned = res.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [
            {"assignee": "Sales Rep", "task": "Follow up with meeting summary and requested collateral", "priority": "High"},
            {"assignee": "Client", "task": "Evaluate internal timeline and budget allocation", "priority": "Medium"}
        ]

    def extract_objections_and_rebuttals(self, transcript):
        """
        Extracts explicit customer objections and generates suggested battlecard rebuttals.
        """
        system_prompt = """
        Identify any customer objections, hesitations, or friction points raised in this call.
        For each objection, provide an effective sales battlecard rebuttal.
        Return ONLY a JSON array of objects:
        [
          {
            "objection": "Description of the objection or hesitation",
            "category": "Pricing" | "Competitor" | "Timing" | "Authority" | "Technical",
            "recommended_rebuttal": "Actionable, persuasive response strategy for the sales rep"
          }
        ]
        Return ONLY raw valid JSON. No markdown code blocks.
        """
        res = self.get_completion(system_prompt, f"Transcript: {transcript}")
        try:
            cleaned = res.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [
            {
                "objection": "General budget and timeline prioritization",
                "category": "Timing",
                "recommended_rebuttal": "Highlight fast time-to-value and offer phased onboarding to reduce upfront commitment friction."
            }
        ]

    def diarize_and_timestamp_transcript(self, raw_transcript):
        """
        Parses raw unstructured transcript into structured conversational turns
        with realistic timestamps, speaker roles, sentiment, and highlights.
        """
        system_prompt = """
        You are a Speech Diarization and Dialogue Structuring Engine.
        Convert the provided call transcript into a sequential turn-by-turn dialogue list.
        Each turn must be attributed to either "Customer" or "Sales Manager" (or "Support Agent").
        Assign realistic incremental timestamps starting from e.g. "13:42".
        
        Return a JSON array of objects:
        [
          {
            "speaker": "Customer" | "Sales Manager",
            "role": "customer" | "agent",
            "timestamp": "13:42",
            "text": "Exact or cleaned line spoken",
            "sentiment": "positive" | "neutral" | "negative",
            "is_highlight": true | false,
            "highlight_tag": "Pain Point" | "Buying Signal" | "Pricing" | "Commitment" | null
          }
        ]
        Return ONLY raw valid JSON. No markdown code blocks.
        """
        res = self.get_completion(system_prompt, f"Raw Transcript:\n{raw_transcript}")
        try:
            cleaned = res.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass
            
        # Fallback manual parsing if LLM JSON parsing fails
        lines = [l.strip() for l in raw_transcript.split('\n') if l.strip()]
        turns = []
        base_minute = 42
        for idx, line in enumerate(lines):
            ts = f"13:{base_minute + (idx // 2):02d}"
            if "customer" in line.lower() or idx % 2 == 0:
                speaker = "Customer"
                role = "customer"
            else:
                speaker = "Sales Manager"
                role = "agent"
            clean_line = line.split(":", 1)[-1].strip() if ":" in line else line
            turns.append({
                "speaker": speaker,
                "role": role,
                "timestamp": ts,
                "text": clean_line,
                "sentiment": "neutral",
                "is_highlight": idx == 0 or "price" in clean_line.lower() or "cost" in clean_line.lower(),
                "highlight_tag": "Pricing" if "price" in clean_line.lower() else ("Key Note" if idx == 0 else None)
            })
        return turns

    def calculate_coaching_metrics(self, turns, overall_sentiment_score=0.7):
        """
        Computes conversational metrics: Talk-to-Listen ratio, Call Quality Score,
        and Closing Probability.
        """
        agent_words = 0
        customer_words = 0
        for turn in turns:
            word_count = len(turn.get("text", "").split())
            if turn.get("role") == "agent" or "sales" in turn.get("speaker", "").lower() or "agent" in turn.get("speaker", "").lower():
                agent_words += word_count
            else:
                customer_words += word_count
                
        total_words = agent_words + customer_words
        if total_words == 0:
            agent_ratio = 54
            customer_ratio = 46
        else:
            agent_ratio = round((agent_words / total_words) * 100)
            customer_ratio = 100 - agent_ratio

        # Quality score (0-100) based on balanced talk ratio and positive sentiment
        ratio_penalty = abs(agent_ratio - 50) # Penalize monologue
        base_quality = int(overall_sentiment_score * 60 + 35 - (ratio_penalty * 0.4))
        quality_score = max(35, min(96, base_quality))
        
        # Closing probability
        close_prob = max(15, min(92, int(quality_score * 0.9 + (10 if overall_sentiment_score > 0.6 else -15))))

        return {
            "agent_talk_pct": agent_ratio,
            "customer_talk_pct": customer_ratio,
            "quality_score": quality_score,
            "closing_probability": close_prob,
            "total_turns": len(turns),
            "estimated_duration_min": max(3, len(turns) * 1.4)
        }

if __name__ == "__main__":
    print("Enhanced LLM Engine ready.")
