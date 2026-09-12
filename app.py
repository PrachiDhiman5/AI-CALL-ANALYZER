import streamlit as st
import os
import time
import json
import tempfile
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Import custom modules
from modules.database import init_db, save_call_analysis, get_all_calls
from modules.data_loader import get_sample_conversations
from modules.preprocessing import clean_transcript
from modules.sentiment import SentimentAnalyzer
from modules.keywords import extract_keywords
from modules.intent import IntentDetector
from modules.embeddings import Embedder
from modules.vector_db import VectorDB
from modules.stt import SpeechToText
from modules.llm_engine import LLMEngine
from modules.insights import InsightsEngine

# Page Configuration
st.set_page_config(
    page_title="CallIQ — Conversational Sales & Support Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
env_api_key = os.getenv("GROQ_API_KEY", "")
if not env_api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    env_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Database quietly
init_db()

# Custom CSS: Warm Cream Canvas, Cobalt Blue & Soft Card Shadows
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-warm: #FDF6ED;
    --bg-warm-card: #FFFFFF;
    --primary-blue: #4667A7;
    --primary-blue-dark: #35528A;
    --accent-gold: #F5BA72;
    --accent-orange: #E89A4B;
    --text-dark: #2D3748;
    --text-muted: #718096;
    --border-subtle: #EFE6DA;
    --shadow-soft: 0 10px 30px rgba(70, 103, 167, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-warm) !important;
    color: var(--text-dark) !important;
}

/* Container Spacing */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}

/* Custom Sidebar Styling (Cobalt Blue matching reference) */
[data-testid="stSidebar"] {
    background-color: #385A9A !important;
    color: #FFFFFF !important;
    border-right: none !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Profile Avatar Box in Sidebar */
.agent-profile-box {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.75rem 0.5rem;
    margin-bottom: 0.75rem;
}

.avatar-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #F5BA72 0%, #E89A4B 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 1.05rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.agent-name-text {
    font-weight: 700;
    font-size: 0.98rem;
    color: #FFFFFF;
    line-height: 1.2;
}

.agent-role-text {
    font-size: 0.76rem;
    color: #D6E4FF;
}

/* Clean Cards */
.studio-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.25rem;
}

/* Metric Cards (Top Row) */
.metric-header-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-soft);
    display: flex;
    align-items: center;
    gap: 1.25rem;
}

.metric-circle-box {
    width: 65px;
    height: 65px;
    border-radius: 50%;
    border: 5px solid #F5BA72;
    border-top-color: #4667A7;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    font-weight: 800;
    color: #4667A7;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}

.metric-label-val {
    font-size: 1.4rem;
    font-weight: 800;
    color: #2D3748;
}

/* Waveform Visualizer */
.waveform-box {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    height: 44px;
    padding: 0 1rem;
}

.wave-bar {
    width: 4px;
    border-radius: 9999px;
    background: #4667A7;
    animation: wavePulse 1.2s ease-in-out infinite alternate;
}

@keyframes wavePulse {
    0% { height: 6px; opacity: 0.35; }
    100% { height: 38px; opacity: 1; }
}

/* Dialogue Turn Bubble */
.dialogue-turn-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.15rem 1.4rem;
    border: 1px solid var(--border-subtle);
    box-shadow: 0 4px 15px rgba(70, 103, 167, 0.03);
    margin-bottom: 0.9rem;
    transition: transform 0.15s, border-color 0.2s;
}

.dialogue-turn-card:hover {
    border-color: #CBD8EC;
    transform: translateY(-1px);
}

.turn-speaker-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}

.speaker-customer {
    font-size: 0.86rem;
    font-weight: 700;
    color: #E89A4B;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.speaker-agent {
    font-size: 0.86rem;
    font-weight: 700;
    color: #4667A7;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.turn-time {
    font-size: 0.74rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

.turn-content {
    font-size: 0.93rem;
    line-height: 1.6;
    color: #2D3748;
}

.highlight-badge {
    background: rgba(245, 186, 114, 0.2);
    border: 1px solid rgba(232, 154, 75, 0.4);
    color: #B25E09;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 6px;
    margin-left: 0.4rem;
}

/* Player Bar Bottom */
.audio-player-toolbar {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 0.85rem 1.4rem;
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-soft);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid var(--border-subtle) !important;
    gap: 1.25rem !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding-bottom: 0.6rem !important;
}

.stTabs [aria-selected="true"] {
    color: var(--primary-blue) !important;
    border-bottom: 3px solid var(--primary-blue) !important;
}

/* Action Items & Objections */
.action-item-box {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 0.9rem;
    border-radius: 12px;
    background: #FFFDF9;
    border: 1px solid var(--border-subtle);
    margin-bottom: 0.55rem;
}

.assignee-chip {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(70, 103, 167, 0.12);
    color: var(--primary-blue);
}

.battlecard-box {
    background: #FFFDF9;
    border-left: 4px solid #E89A4B;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.75rem;
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
}

/* Buttons */
.stButton button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Default Demo Call Data (Instantly loaded on startup for zero wait time)
DEFAULT_DEMO_DATA = {
    "intent": "Pricing Objection & Enterprise Migration",
    "risk_level": "🟢 High-Conviction Opportunity",
    "sentiment_label": "Positive",
    "sentiment_score": 0.84,
    "keywords": ["enterprise onboarding", "implementation cost", "data migration", "SOC2 compliance", "Q3 budget", "contract addendum"],
    "coaching_metrics": {
        "quality_score": 76,
        "agent_talk_pct": 48,
        "customer_talk_pct": 52,
        "estimated_duration_min": 13.9,
        "total_turns": 6,
        "closing_probability": 85
    },
    "dialogue_turns": [
        {
            "speaker": "Customer",
            "role": "customer",
            "timestamp": "13:43",
            "text": "And what we basically wanted to do was we wanted to take all the good things that were working for these brands into consideration while designing our website.",
            "is_highlight": False
        },
        {
            "speaker": "Sales Manager",
            "role": "agent",
            "timestamp": "13:44",
            "text": "We understand, and what is the main goal of your brand for this upcoming quarter?",
            "is_highlight": False
        },
        {
            "speaker": "Customer",
            "role": "customer",
            "timestamp": "13:47",
            "text": "Our main goal is to provide a quality product to our enterprise customers. We are interested in their success, but the $25,000 upfront implementation cost is giving our finance committee pause.",
            "is_highlight": True,
            "highlight_tag": "Pricing Barrier"
        },
        {
            "speaker": "Sales Manager",
            "role": "agent",
            "timestamp": "13:51",
            "text": "Budget predictability is crucial. What if we amortize that onboarding cost across a 2-year term and include dedicated migration engineers at no additional retainer?",
            "is_highlight": True,
            "highlight_tag": "Rebuttal Catalyst"
        },
        {
            "speaker": "Customer",
            "role": "customer",
            "timestamp": "13:55",
            "text": "That would resolve our committee's hesitation. If you can email the SOC2 compliance packet and updated addendum by Thursday, we will execute.",
            "is_highlight": True,
            "highlight_tag": "Buying Signal"
        },
        {
            "speaker": "Sales Manager",
            "role": "agent",
            "timestamp": "13:59",
            "text": "I will personally deliver the security packet and addendum to your inbox by 4 PM today. Looking forward to our partnership!",
            "is_highlight": False
        }
    ],
    "objections": [
        {
            "objection": "$25,000 Upfront Implementation & Onboarding Cost",
            "category": "Pricing",
            "recommended_rebuttal": "Offer 2-year term amortization and highlight that dedicated data migration & SSO setup eliminate 80+ hours of internal engineering overhead."
        },
        {
            "objection": "Security & Regulatory Compliance Approval",
            "category": "Authority",
            "recommended_rebuttal": "Provide pre-packaged SOC2 Type II, ISO 27001, and GDPR documentation packet directly to their InfoSec director."
        }
    ],
    "action_items": [
        {"assignee": "Sales Manager", "task": "Email SOC2 compliance certification and security whitepaper by 4 PM", "priority": "High"},
        {"assignee": "Sales Manager", "task": "Draft and deliver updated 2-year amortized contract addendum", "priority": "High"},
        {"assignee": "Customer", "task": "Review security documentation with internal InfoSec committee before Thursday", "priority": "Medium"}
    ],
    "insights": """**Executive Summary:**
Customer validated enterprise fit but raised friction regarding upfront onboarding expense. Representative successfully neutralized objection with 2-year amortization structure and secured commitment to sign upon security review.

**Key Buying Drivers:**
- Customer prioritization of product quality and dedicated migration support.
- Direct alignment on compliance requirements and fast-track implementation.

**Sales Coaching Recommendations:**
- Excellent active listening and swift objection handling without discounting core recurring ARR.
- Follow up immediately with security documentation to maintain deal momentum."""
}

# Session State Initialization
if "active_call_data" not in st.session_state:
    st.session_state["active_call_data"] = DEFAULT_DEMO_DATA

# Sidebar Navigation (Matching Reference Image)
with st.sidebar:
    st.markdown("""
    <div class="agent-profile-box">
        <div class="avatar-circle">AP</div>
        <div>
            <div class="agent-name-text">Alexander P.</div>
            <div class="agent-role-text">Sales Manager</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    view_mode = st.radio(
        "Navigation",
        ["🎙️ Live Dialogue Studio", "📁 Call Records Archive", "📊 Conversation Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚡ System Status")
    st.markdown("**Core Engine:** `Groq GPT-OSS / Qwen`")
    st.markdown("**Audio Engine:** `Whisper-Large-V3`")
    
    if env_api_key:
        st.success("● Cloud LLM Connected")
    else:
        st.warning("⚠️ GROQ_API_KEY missing from .env")

# =============================================================================
# VIEW 1: LIVE DIALOGUE STUDIO (MATCHING THE REFERENCE IMAGE EXACTLY)
# =============================================================================
if view_mode == "🎙️ Live Dialogue Studio":

    # --- TOP ACTION / INGESTION EXPANDER ---
    with st.expander("📥 Ingest New Call (Audio Upload or Paste Transcript)", expanded=False):
        col_in1, col_in2 = st.columns([1.2, 1])
        with col_in1:
            raw_input = st.text_area(
                "Transcript Input",
                height=120,
                placeholder="Paste conversational dialogue here...\nCustomer: We need to discuss pricing...\nSales Rep: Sure, let's explore..."
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🚀 Analyze Transcript", type="primary"):
                    if raw_input.strip():
                        with st.spinner("Processing dialogue intelligence..."):
                            llm = LLMEngine(env_api_key)
                            insights_engine = InsightsEngine(
                                SentimentAnalyzer(), IntentDetector(llm), extract_keywords,
                                Embedder(), VectorDB(), llm
                            )
                            cleaned = clean_transcript(raw_input)
                            res = insights_engine.run_full_analysis(cleaned, filename="Manual_Input", source_type="Text")
                            save_call_analysis(res)
                            st.session_state["active_call_data"] = res
                            st.rerun()
            with col_b2:
                if st.button("✨ Load Sample Call"):
                    samples = get_sample_conversations(1)
                    if samples:
                        with st.spinner("Analyzing sample..."):
                            llm = LLMEngine(env_api_key)
                            insights_engine = InsightsEngine(
                                SentimentAnalyzer(), IntentDetector(llm), extract_keywords,
                                Embedder(), VectorDB(), llm
                            )
                            res = insights_engine.run_full_analysis(samples[0]["dialogue"], filename="Sample_Call", source_type="Sample")
                            save_call_analysis(res)
                            st.session_state["active_call_data"] = res
                            st.rerun()

        with col_in2:
            audio_file = st.file_uploader("Upload Call Audio (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
            if audio_file and st.button("🎙️ Transcribe & Analyze Audio", type="primary"):
                if not env_api_key:
                    st.error("GROQ_API_KEY required for audio transcription.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                        tmp.write(audio_file.getvalue())
                        tmp_path = tmp.name
                    with st.spinner("Transcribing audio using Whisper-Large-V3..."):
                        transcription, _ = SpeechToText(env_api_key).transcribe(tmp_path)
                    os.remove(tmp_path)
                    with st.spinner("Analyzing transcribed dialogue..."):
                        llm = LLMEngine(env_api_key)
                        insights_engine = InsightsEngine(
                            SentimentAnalyzer(), IntentDetector(llm), extract_keywords,
                            Embedder(), VectorDB(), llm
                        )
                        cleaned = clean_transcript(transcription)
                        res = insights_engine.run_full_analysis(cleaned, filename=audio_file.name, source_type="Audio")
                        save_call_analysis(res)
                        st.session_state["active_call_data"] = res
                        st.rerun()

    # Active Call Data
    data = st.session_state["active_call_data"]
    coaching = data.get("coaching_metrics", {})
    turns = data.get("dialogue_turns", [])

    # --- TOP 3 CARDS ROW (MATCHING REFERENCE IMAGE) ---
    col_t1, col_t2, col_t3 = st.columns([1.1, 1.1, 1.4])

    with col_t1:
        st.markdown(f"""
        <div class="metric-header-card">
            <div class="metric-circle-box">
                {coaching.get('quality_score', 76)}%
            </div>
            <div>
                <div class="metric-label-title">Call Quality Score</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">
                    <div>● Talk Ratio: <strong style="color: #4667A7;">{coaching.get('agent_talk_pct', 48)}%</strong></div>
                    <div>● Listen Ratio: <strong style="color: #E89A4B;">{coaching.get('customer_talk_pct', 52)}%</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
        <div class="metric-header-card">
            <div style="font-size: 2rem; color: #4667A7;">📊</div>
            <div>
                <div class="metric-label-title">Minutes Processed</div>
                <div class="metric-label-val">{coaching.get('estimated_duration_min', 13.9):.1f}m</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">{len(turns)} Dialogue Turns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        st.markdown(f"""
        <div class="metric-header-card">
            <div>
                <div class="metric-label-title">Primary Intent & Health</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.3rem;">
                    {data.get('intent', 'Enterprise Call')}
                </div>
                <span style="background: rgba(70, 103, 167, 0.1); color: #4667A7; font-weight: 700; font-size: 0.8rem; padding: 0.25rem 0.65rem; border-radius: 6px;">
                    {data.get('risk_level', '🟢 Healthy Opportunity')}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # --- MAIN SPLIT LAYOUT: LEFT TRANSCRIPT + RIGHT COACHING SUITE ---
    col_left_stream, col_right_suite = st.columns([1.45, 1.1])

    # Left Column: Waveform & Dialogue Stream
    with col_left_stream:
        # Waveform Box
        st.markdown("""
        <div class="studio-card" style="padding: 1rem 1.25rem; margin-bottom: 0.85rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Acoustic Waveform Analysis</span>
                <span style="font-size: 0.74rem; color: #4667A7; font-weight: 600;">Dual Channel (16 kHz)</span>
            </div>
            <div class="waveform-box">
                <div class="wave-bar" style="height: 12px; animation-delay: 0.1s;"></div>
                <div class="wave-bar" style="height: 24px; animation-delay: 0.3s; background: #F5BA72;"></div>
                <div class="wave-bar" style="height: 36px; animation-delay: 0.2s;"></div>
                <div class="wave-bar" style="height: 16px; animation-delay: 0.5s;"></div>
                <div class="wave-bar" style="height: 40px; animation-delay: 0.4s; background: #E89A4B;"></div>
                <div class="wave-bar" style="height: 28px; animation-delay: 0.1s;"></div>
                <div class="wave-bar" style="height: 18px; animation-delay: 0.6s;"></div>
                <div class="wave-bar" style="height: 34px; animation-delay: 0.2s;"></div>
                <div class="wave-bar" style="height: 14px; animation-delay: 0.3s; background: #F5BA72;"></div>
                <div class="wave-bar" style="height: 26px; animation-delay: 0.5s;"></div>
                <div class="wave-bar" style="height: 38px; animation-delay: 0.1s;"></div>
                <div class="wave-bar" style="height: 22px; animation-delay: 0.4s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Transcript Search Filter
        search_kw = st.text_input("Search transcript...", placeholder="🔍 Search transcript keywords or speaker...", label_visibility="collapsed")

        # Filtered Turns
        filtered_turns = [
            t for t in turns 
            if not search_kw or search_kw.lower() in t.get("text", "").lower() or search_kw.lower() in t.get("speaker", "").lower()
        ]

        if filtered_turns:
            for idx, turn in enumerate(filtered_turns):
                is_cust = "cust" in turn.get("speaker", "").lower() or turn.get("role") == "customer"
                speaker_class = "speaker-customer" if is_cust else "speaker-agent"
                icon = "👤" if is_cust else "💼"
                
                badge_html = f"<span class='highlight-badge'>{turn.get('highlight_tag')}</span>" if turn.get("is_highlight") and turn.get("highlight_tag") else ""
                
                st.markdown(f"""
                <div class="dialogue-turn-card">
                    <div class="turn-speaker-row">
                        <span class="{speaker_class}">
                            <span>{icon}</span>
                            <span>{turn.get('speaker', 'Speaker')}</span>
                            {badge_html}
                        </span>
                        <span class="turn-time">{turn.get('timestamp', f'13:{40+idx}')}</span>
                    </div>
                    <div class="turn-content">{turn.get('text', '')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No dialogue turns match your keyword filter.")

        # Audio Controls Footer
        st.markdown("""
        <div class="audio-player-toolbar">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="cursor: pointer; font-size: 1.1rem;">⏮</span>
                <span style="cursor: pointer; font-size: 1.3rem; color: #4667A7;">▶</span>
                <span style="cursor: pointer; font-size: 1.1rem;">⏭</span>
                <span style="font-size: 0.8rem; font-family: 'JetBrains Mono'; color: var(--text-muted); margin-left: 0.4rem;">13:43 / 18:14</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 0.76rem; font-weight: 700; background: #FDF6ED; padding: 0.2rem 0.5rem; border-radius: 6px;">1.0x Speed</span>
                <span style="cursor: pointer; font-size: 1rem;">🔊</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Right Column: Coaching & Intelligence Suite
    with col_right_suite:
        st.markdown("### 💡 Intelligence Suite")
        
        tab_hl, tab_obj, tab_act, tab_strat = st.tabs([
            "✨ Highlights", "🛡️ Objections", "📋 Action Items", "📊 Strategy"
        ])

        with tab_hl:
            st.markdown("#### Primary Topic Tags")
            kw_chips = " ".join([f"<span style='background: #EFF3FB; color: #35528A; padding: 0.25rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; margin-right: 0.35rem; display: inline-block; margin-bottom: 0.35rem;'>#{k}</span>" for k in data.get('keywords', [])])
            st.markdown(kw_chips, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("#### Key Conversation Excerpts")
            hl_turns = [t for t in turns if t.get('is_highlight')]
            if hl_turns:
                for ht in hl_turns:
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border-left: 3px solid #E89A4B; padding: 0.65rem 0.9rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; font-size: 0.88rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                        <strong style="color: #E89A4B;">{ht.get('speaker')}:</strong> "{ht.get('text')}"
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No distinct highlight quotes tagged.")

        with tab_obj:
            st.markdown("#### Customer Friction Points & Rebuttals")
            objs = data.get('objections', [])
            if objs:
                for o in objs:
                    st.markdown(f"""
                    <div class="battlecard-box">
                        <div style="font-weight: 700; font-size: 0.92rem; color: #2D3748; margin-bottom: 0.3rem;">
                            ⚠️ {o.get('objection', 'Objection')} <span style="font-size: 0.74rem; color: #718096;">({o.get('category', 'Sales')})</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #4A5568; line-height: 1.5;">
                            <strong>Recommended Strategy:</strong> {o.get('recommended_rebuttal', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✓ No severe customer objections identified.")

        with tab_act:
            st.markdown("#### Post-Call Commitments")
            acts = data.get('action_items', [])
            if acts:
                for a in acts:
                    p_col = "#EF4444" if a.get('priority') == 'High' else ("#F59E0B" if a.get('priority') == 'Medium' else "#10B981")
                    st.markdown(f"""
                    <div class="action-item-box">
                        <input type="checkbox" style="margin-top: 0.2rem;">
                        <div>
                            <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.2rem;">
                                <span class="assignee-chip">{a.get('assignee', 'Rep')}</span>
                                <span style="font-size: 0.72rem; font-weight: 700; color: {p_col};">● {a.get('priority', 'Normal')} Priority</span>
                            </div>
                            <div style="font-size: 0.88rem; color: #2D3748;">{a.get('task', '')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific follow-up tasks detected.")

        with tab_strat:
            st.markdown(f"""
            <div class="studio-card" style="font-size: 0.9rem; line-height: 1.6;">
                {data.get('insights', '').replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# VIEW 2: CALL RECORDS ARCHIVE
# =============================================================================
elif view_mode == "📁 Call Records Archive":
    st.markdown("## 📁 Call Records & CRM History")
    st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem;'>Browse historical call audits, deal classifications, and past conversational transcripts.</p>", unsafe_allow_html=True)
    
    calls = get_all_calls()
    if not calls:
        st.info("No archived calls found in database yet.")
    else:
        df = pd.DataFrame([dict(c) for c in calls])
        
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            q = st.text_input("Search CRM records...", placeholder="Search intent, customer dialogue, or keywords...")
        with col_s2:
            s_filter = st.selectbox("Sentiment Filter", ["All", "Positive", "Neutral", "Negative"])
            
        filtered = df
        if s_filter != "All":
            filtered = filtered[filtered["sentiment_label"] == s_filter]
        if q:
            filtered = filtered[filtered["transcript"].str.contains(q, case=False, na=False) | filtered["intent"].str.contains(q, case=False, na=False)]
            
        st.markdown(f"**Showing {len(filtered)} recorded calls**")
        for idx, row in filtered.iterrows():
            with st.expander(f"📞 {row.get('intent', 'Call Interaction')} — {row.get('timestamp', '')[:16]} ({row.get('risk_level', 'Healthy')})"):
                st.markdown(f"**Sentiment:** `{row.get('sentiment_label')}` | **Score:** `{row.get('sentiment_score', 0):.2f}`")
                st.markdown(f"**Transcript Excerpt:**\n> {row.get('transcript', '')[:280]}...")
                if st.button(f"🔍 Load Call #{row.get('id')} in Live Studio", key=f"rec_{row.get('id')}"):
                    # Load turns
                    try:
                        turns_json = json.loads(row.get('dialogue_turns', '[]'))
                    except Exception:
                        turns_json = []
                    try:
                        actions_json = json.loads(row.get('action_items', '[]'))
                    except Exception:
                        actions_json = []
                    try:
                        objs_json = json.loads(row.get('objections', '[]'))
                    except Exception:
                        objs_json = []
                    try:
                        coaching_json = json.loads(row.get('coaching_metrics', '{}'))
                    except Exception:
                        coaching_json = {}
                        
                    st.session_state["active_call_data"] = {
                        "intent": row.get("intent"),
                        "risk_level": row.get("risk_level"),
                        "sentiment_label": row.get("sentiment_label"),
                        "sentiment_score": row.get("sentiment_score"),
                        "keywords": json.loads(row.get("keywords", "[]")),
                        "insights": row.get("insights"),
                        "dialogue_turns": turns_json if turns_json else DEFAULT_DEMO_DATA["dialogue_turns"],
                        "action_items": actions_json if actions_json else DEFAULT_DEMO_DATA["action_items"],
                        "objections": objs_json if objs_json else DEFAULT_DEMO_DATA["objections"],
                        "coaching_metrics": coaching_json if coaching_json else DEFAULT_DEMO_DATA["coaching_metrics"]
                    }
                    st.rerun()

# =============================================================================
# VIEW 3: CONVERSATION ANALYTICS
# =============================================================================
elif view_mode == "📊 Conversation Analytics":
    st.markdown("## 📊 Team Performance & Conversation Analytics")
    st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem;'>Macro intent trends, customer friction patterns, and sentiment health over time.</p>", unsafe_allow_html=True)
    
    calls = get_all_calls()
    if not calls:
        st.info("Analyze calls to populate aggregate team analytics.")
    else:
        df = pd.DataFrame([dict(c) for c in calls])
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            fig_intent = px.pie(
                df,
                names="intent",
                title="Customer Intent & Friction Categorization",
                hole=0.55,
                color_discrete_sequence=["#4667A7", "#F5BA72", "#E89A4B", "#6282BF", "#8EA7D1"]
            )
            fig_intent.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Plus Jakarta Sans", "color": "#2D3748"},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_intent, use_container_width=True)

        with col_a2:
            df["date"] = pd.to_datetime(df["timestamp"])
            fig_trend = px.line(
                df.sort_values("date"),
                x="date",
                y="sentiment_score",
                title="Sentiment Health Trajectory",
                markers=True,
                color_discrete_sequence=["#4667A7"]
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Plus Jakarta Sans", "color": "#2D3748"},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor="#EFE6DA")
            st.plotly_chart(fig_trend, use_container_width=True)
