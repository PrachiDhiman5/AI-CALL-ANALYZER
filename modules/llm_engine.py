from groq import Groq
import time

class LLMEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
        self.model = "llama-3.3-70b-versatile"

    def get_completion(self, system_prompt, user_prompt):
        if not self.client:
            return "Error: API Key missing."
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.5,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def get_intent(self, transcript):
        system_prompt = """
        Analyze the call transcript and identify the SPECIFIC primary intent of the customer.
        NEVER return generic categories like 'Sales Inquiry' or 'Inquiry'.
        Be highly specific and differentiate between interest and rejection.
        Use detailed business categories such as: 
        - Pricing Objection / Cost Value Gap (Specifically for complaints about price)
        - Disinterest / Lead Lost (Specifically when customer says 'No' or 'Not interested')
        - Competitor Comparison (e.g., Salesforce vs HubSpot)
        - Technical Troubleshooting (Specific Problem)
        - Product Feature Inquiry
        - Qualified Sales Prospecting (Customer wants to buy or sign up)
        - Subscription Cancellation / Churn Risk
        - Refund Eligibility Check
        
        If the customer is rejecting an offer due to price, ALWAYS include the word 'Objection' in your response.
        Return ONLY the 2-4 word category name. No punctuation.
        """
        res = self.get_completion(system_prompt, f"Transcript: {transcript}")
        return res.strip().replace(".", "")

    def extract_business_keywords(self, transcript):
        """
        Extracts meaningful business keywords using LLM intelligence.
        """
        system_prompt = "Extract 5-8 highly relevant business keywords or phrases from this call transcript. Focus on pain points, products, and customer needs. Return them as a comma-separated list."
        res = self.get_completion(system_prompt, f"Transcript: {transcript}")
        keywords = [k.strip() for k in res.split(',')]
        return keywords[:8]

    def generate_insights(self, transcript, sentiment, keywords, rag_context=""):
        system_prompt = """
        You are a Production-Grade AI Call Analyzer. Your goal is to generate actionable business insights from a call transcript.
        Consider:
        1. Customer Sentiment
        2. Key Themes/Keywords
        3. Historical Context (if provided)
        
        Provide your response in Markdown with these sections:
        - **Summary**: 2-sentence overview.
        - **Key Issues**: Bullets.
        - **Actionable Recommendation**: What should the company do next?
        """
        
        user_prompt = f"""
        Transcript: {transcript}
        Sentiment: {sentiment}
        Keywords: {', '.join(keywords)}
        Historical Context from similar calls: {rag_context}
        """
        
        return self.get_completion(system_prompt, user_prompt)

if __name__ == "__main__":
    print("LLM Engine ready.")
