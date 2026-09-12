import pandas as pd
import random

BUILTIN_SAMPLES = [
    {
        "dialogue": "Customer: Hi Alexander, thanks for jumping on. We are actively reviewing your proposal for the enterprise tier, but the upfront implementation cost of $25,000 seems high compared to our current provider.\nSales Manager: I completely understand, Sarah. Budget predictability is key. Let me clarify what that covers: dedicated data migration, custom SSO integration, and 24/7 SLA support. If we amortize that over a 2-year term, we can waive the onboarding retainer.\nCustomer: That would significantly help our Q3 budget approval. Can you also send over the compliance certificates for SOC2 and GDPR?\nSales Manager: Absolutely. I'll email the security packet and the updated contract addendum by end of day today.\nCustomer: Excellent. If the security team signs off, we should be ready to execute by Friday.",
        "summary": "Customer raised pricing objection regarding onboarding costs; Sales Manager offered 2-year amortization and committed to sending security compliance packet by EOD."
    },
    {
        "dialogue": "Customer: Hello, I am calling because our production API endpoint has been throwing 504 gateway timeout errors for the past 45 minutes.\nSales Manager: I am very sorry to hear that. Let me look up your account ID immediately. Can you confirm the service region?\nCustomer: US-East-1. Our mobile checkout is completely blocked right now.\nSales Manager: I see the alert on our infrastructure board. Our DevOps team is actively rolling out a hotfix to routing node 4. I am assigning highest priority ticket #9821 and will stay on the line until we see latency normalize.\nCustomer: Thank you, I appreciate the quick escalation.",
        "summary": "Urgent technical outage in US-East-1 affecting customer mobile checkout. Support rep escalated to DevOps with ticket #9821."
    },
    {
        "dialogue": "Customer: We evaluated both your platform and Competitor X. Their automated workflow builder has drag-and-drop webhooks which our marketing ops team really liked.\nSales Manager: Competitor X has a good UI for basic triggers, but where our clients see 3x ROI is in native bi-directional CRM syncing and automated attribution modeling without third-party connectors.\nCustomer: Interesting. Can we schedule a 20-minute technical walkthrough with our lead architect next Tuesday?\nSales Manager: I'd be delighted to host that. I will send a calendar invite for Tuesday at 2 PM with our solutions engineer.",
        "summary": "Competitive evaluation against Competitor X; Sales Manager highlighted bi-directional sync advantage and booked technical demo for Tuesday."
    },
    {
        "dialogue": "Customer: I am calling to discuss renewing our annual subscription, but our team headcount downsized by 30% recently.\nSales Manager: Thank you for your transparency, Michael. We want to ensure our partnership scales with your team's current structure. We can adjust your license count from 50 to 35 seats without forfeiting your volume discount.\nCustomer: That is a huge relief. We definitely want to keep using the analytics dashboard.\nSales Manager: Fantastic. I will generate the revised renewal agreement with the updated seat tier and send it over for e-signature.",
        "summary": "Customer requested seat reduction due to team downsizing; Sales Manager retained account by adjusting seats to 35 while preserving volume discounts."
    }
]

def load_hf_dataset():
    """
    Loads dataset from Hugging Face if available, otherwise returns rich built-in samples.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("knkarthick/dialogsum", split="train")
        return pd.DataFrame(dataset)
    except Exception as e:
        print(f"Notice: Loading built-in conversational dataset ({e}).")
        return pd.DataFrame(BUILTIN_SAMPLES)

def get_sample_conversations(n=4):
    """
    Returns n sample conversations.
    """
    try:
        df = load_hf_dataset()
        if not df.empty:
            sample_count = min(n, len(df))
            return df.sample(sample_count).to_dict('records')
    except Exception:
        pass
    return random.sample(BUILTIN_SAMPLES, min(n, len(BUILTIN_SAMPLES)))

if __name__ == "__main__":
    samples = get_sample_conversations(2)
    for i, s in enumerate(samples):
        print(f"Sample {i+1}:\n{s['dialogue']}\n")
