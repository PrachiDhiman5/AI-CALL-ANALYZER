import streamlit as st
import os
import time
import json
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import custom modules
from modules.database import init_db, save_call_analysis, get_all_calls
from modules.data_loader import get_sample_conversations, load_hf_dataset
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

# Custom CSS: Warm Cream, Cobalt Blue & Slate Executive Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-warm: #FDF6ED;
    --bg-warm-light: #FFF9F2;
    --bg-card: #FFFFFF;
    --primary-blue: #4667A7;
    --primary-blue-dark: #36538E;
    --accent-gold: #F5BA72;
    --accent-orange: #E89A4B;
    --text-dark: #323232;
    --text-muted: #7E8B9B;
    --border-light: #EFE6DA;
    --shadow-soft: 0 10px 30px rgba(70, 103, 167, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-warm) !important;
    color: var(--text-dark) !important;
}

/* Remove default padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1340px !important;
}

/* Custom Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #3B5B99 !important;
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
    margin-bottom: 1rem;
}

.avatar-circle {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, #F5BA72 0%, #E89A4B 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.agent-name-text {
    font-weight: 700;
    font-size: 1rem;
    color: #FFFFFF;
    line-height: 1.2;
}

.agent-role-text {
    font-size: 0.78rem;
    color: #CFE0FC;
}

/* Sidebar Nav Buttons */
.stButton button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

/* White Dashboard Cards */
.dash-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid var(--border-light);
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.25rem;
}

/* Metric Pill Ring */
.metric-ring-card {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    border: 1px solid var(--border-light);
    box-shadow: var(--shadow-soft);
}

.metric-ring-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--primary-blue);
    line-height: 1;
}

.metric-ring-lbl {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.05em;
}

/* Waveform Bars Simulator */
.waveform-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    height: 48px;
    padding: 0 1rem;
}

.waveform-bar {
    width: 4px;
    border-radius: 9999px;
    background: var(--primary-blue);
    animation: waveGlow 1.4s ease-in-out infinite alternate;
}

@keyframes waveGlow {
    0% { height: 8px; opacity: 0.4; }
    100% { height: 42px; opacity: 1; }
}

/* Turn Dialogue Card */
.turn-bubble {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    border: 1px solid var(--border-light);
    box-shadow: 0 4px 15px rgba(70, 103, 167, 0.04);
    margin-bottom: 1rem;
    position: relative;
    transition: transform 0.15s, border-color 0.2s;
}

.turn-bubble:hover {
    border-color: #CCD8ED;
    transform: translateY(-1px);
}

.speaker-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.speaker-tag-customer {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-orange);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.speaker-tag-agent {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary-blue);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.turn-timestamp {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

.turn-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-dark);
}

.highlight-chip {
    display: inline-block;
    background: rgba(245, 186, 114, 0.2);
    border: 1px solid rgba(232, 154, 75, 0.35);
    color: #B2620A;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    margin-left: 0.5rem;
}

/* Audio Player Bar */
.player-bar {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 0.85rem 1.5rem;
    border: 1px solid var(--border-light);
    box-shadow: var(--shadow-soft);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid var(--border-light) !important;
    gap: 1.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding-bottom: 0.75rem !important;
}

.stTabs [aria-selected="true"] {
    color: var(--primary-blue) !important;
    border-bottom: 3px solid var(--primary-blue) !important;
}

/* Action Items Card */
.task-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: var(--bg-warm-light);
    border: 1px solid var(--border-light);
    margin-bottom: 0.6rem;
}

.task-assignee {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: rgba(70, 103, 167, 0.12);
    color: var(--primary-blue);
}

.task-desc {
    font-size: 0.88rem;
    color: var(--text-dark);
    line-height: 1.4;
}

/* Objection Card */
.objection-card {
    background: var(--bg-warm-light);
    border-left: 4px solid var(--accent-orange);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
}

.objection-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--text-dark);
    margin-bottom: 0.35rem;
}

.rebuttal-box {
    font-size: 0.85rem;
    color: #4B5563;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# Initialize Core Models
@st.cache_resource
def load_core_engines():
    sentiment = SentimentAnalyzer()
    embedder = Embedder()
    vector_db = VectorDB()
    intent = IntentDetector()
    init_db()
    return {
        "sentiment": sentiment,
        "embedder": embedder,
        "vector_db": vector_db,
        "intent": intent
    }

with st.spinner("Initializing Conversational Intelligence Engine..."):
    models = load_core_engines()

# Sidebar Setup (Matching Reference Image)
with st.sidebar:
    st.markdown("""
    <div class="agent-profile-box">
        <div class="avatar-circle">AP</div>
        <div>
            <div class="agent-name-text">Alexander P.</div>
            <div class="agent-role-text">Senior Sales Director</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav_selection = st.radio(
        "Navigation",
        ["🎙️ Live Call Analyzer", "📁 Call Records Archive", "📊 Coaching & Analytics", "📚 Reference Library"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Telemetry")
    st.markdown("**LLM Cluster:** `Groq GPT-OSS / Qwen`")
    st.markdown("**STT Core:** `Whisper-Large-V3`")
    st.markdown("**Vector Store:** `FAISS Neural Index`")
    
    if not env_api_key:
        st.warning("⚠️ `GROQ_API_KEY` missing from .env")
    else:
        st.success("● API Cloud Connected")

    st.markdown("---")
    if st.button("🔄 Clear Active Session"):
        st.session_state.clear()
        st.rerun()

# =============================================================================
# VIEW 1: LIVE CALL ANALYZER (THE CENTERPIECE MATCHING REFERENCE IMAGE)
# =============================================================================
if nav_selection == "🎙️ Live Call Analyzer":
    
    # Check if there is pre-loaded sample text from Explorer
    initial_text = st.session_state.get('active_transcript', "")
    
    st.markdown("## 🎙️ Call Intelligence & Dialogue Studio")
    st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem; margin-top: -0.5rem;'>Interactive speech diarization, objection handling, and customer sentiment analytics.</p>", unsafe_allow_html=True)
    
    # Top Ingestion Row
    with st.expander("📥 Ingest Call Audio or Transcript", expanded=not bool(st.session_state.get('analysis_result'))):
        tab_text, tab_audio = st.tabs(["📝 Text Transcript", "🎵 Audio Recording (.mp3 / .wav)"])
        
        with tab_text:
            transcript_input = st.text_area(
                "Paste Call Transcript",
                value=initial_text,
                height=140,
                placeholder="Customer: Hi, we are reviewing your enterprise pricing...\nSales Manager: Thanks for connecting! Let me walk you through..."
            )
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                run_text_btn = st.button("🚀 Analyze Call", type="primary")
            with col_btn2:
                if st.button("✨ Load Sample SaaS Call"):
                    sample_calls = get_sample_conversations(1)
                    if sample_calls:
                        st.session_state['active_transcript'] = sample_calls[0]['dialogue']
                        st.rerun()

        with tab_audio:
            uploaded_audio = st.file_uploader("Upload Call Audio File", type=["mp3", "wav", "m4a"])
            if uploaded_audio:
                st.audio(uploaded_audio)
                if st.button("🎙️ Transcribe & Run Full Audit", type="primary"):
                    if not env_api_key:
                        st.error("GROQ_API_KEY is required for audio transcription.")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp:
                            tmp.write(uploaded_audio.getvalue())
                            tmp_path = tmp.name
                        with st.spinner("Transcribing audio using Whisper-Large-V3..."):
                            transcribed_text, _ = SpeechToText(env_api_key).transcribe(tmp_path)
                        os.remove(tmp_path)
                        st.session_state['active_transcript'] = transcribed_text
                        st.session_state['run_pipeline_trigger'] = True
                        st.rerun()

    # Trigger Pipeline Execution
    if run_text_btn and transcript_input.strip():
        st.session_state['active_transcript'] = transcript_input.strip()
        st.session_state['run_pipeline_trigger'] = True

    if st.session_state.get('run_pipeline_trigger') and st.session_state.get('active_transcript'):
        st.session_state['run_pipeline_trigger'] = False
        raw_text = st.session_state['active_transcript']
        
        with st.status("🧠 Running Multi-Stage Conversational Intelligence...", expanded=True) as status:
            st.write("1. Normalizing dialogue syntax & vocabulary...")
            cleaned = clean_transcript(raw_text)
            
            st.write("2. Computing sentiment polarity and emotional valence...")
            llm = LLMEngine(env_api_key)
            insights_engine = InsightsEngine(
                models['sentiment'], models['intent'], extract_keywords,
                models['embedder'], models['vector_db'], llm
            )
            
            st.write("3. Diarizing turns, detecting objections & calculating talk-ratio...")
            results = insights_engine.run_full_analysis(cleaned, filename="Live_Studio_Input", source_type="Studio")
            
            st.write("4. Archiving report to persistent database...")
            save_call_analysis(results)
            status.update(label="✓ Analysis Complete & Rendered", state="complete", expanded=False)
            
            st.session_state['analysis_result'] = results

    # =========================================================================
    # RENDER THE COMPLETE STUDIO CANVAS (MATCHING REFERENCE IMAGE)
    # =========================================================================
    res = st.session_state.get('analysis_result')
    
    if res:
        coaching = res.get('coaching_metrics', {})
        turns = res.get('dialogue_turns', [])
        
        # 1. TOP METRIC ROW (MATCHING REFERENCE IMAGE)
        col_m1, col_m2, col_m3 = st.columns([1.2, 1.2, 1.6])
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-ring-card">
                <div>
                    <div class="metric-ring-val">{coaching.get('quality_score', 82)}%</div>
                    <div class="metric-ring-lbl">Call Quality Score</div>
                </div>
                <div style="font-size: 0.78rem; color: var(--text-muted); border-left: 1px solid var(--border-light); padding-left: 0.75rem;">
                    <div>● Talk Ratio: <strong style="color: var(--primary-blue);">{coaching.get('agent_talk_pct', 52)}%</strong></div>
                    <div>● Sentiment: <strong style="color: #10B981;">{res.get('sentiment_label', 'Positive')}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-ring-card">
                <div>
                    <div class="metric-ring-val">{coaching.get('estimated_duration_min', 13.5):.1f}m</div>
                    <div class="metric-ring-lbl">Processed Duration</div>
                </div>
                <div style="font-size: 0.78rem; color: var(--text-muted); border-left: 1px solid var(--border-light); padding-left: 0.75rem;">
                    <div>● Total Turns: <strong>{coaching.get('total_turns', len(turns))}</strong></div>
                    <div>● Latency: <strong>{res.get('latency_metrics', {}).get('total_latency', 450):.0f}ms</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="metric-ring-card">
                <div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.25rem;">
                        {res.get('intent', 'Enterprise Sales Prospect')}
                    </div>
                    <div class="metric-ring-lbl">Primary Intent & Health</div>
                </div>
                <div style="margin-left: auto;">
                    <span style="background: rgba(70, 103, 167, 0.12); color: var(--primary-blue); font-weight: 700; font-size: 0.8rem; padding: 0.35rem 0.75rem; border-radius: 8px;">
                        {res.get('risk_level', '🟢 Healthy Deal')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # 2. MAIN SPLIT CANVAS: LEFT DIALOGUE + RIGHT COACHING SUITE
        col_left_canvas, col_right_suite = st.columns([1.5, 1.1])

        # --- LEFT CANVAS: WAVEFORM & DIARIZED TRANSCRIPT ---
        with col_left_canvas:
            st.markdown("### 🎙️ Dialogue Audio & Turn Breakdown")
            
            # Simulated Interactive Waveform Player Box
            st.markdown("""
            <div class="dash-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Acoustic Waveform Analysis</span>
                    <span style="font-size: 0.75rem; color: var(--primary-blue); font-weight: 600;">Stereo Channel (16 kHz)</span>
                </div>
                <div class="waveform-container">
                    <div class="waveform-bar" style="height: 14px; animation-delay: 0.1s;"></div>
                    <div class="waveform-bar" style="height: 28px; animation-delay: 0.3s; background: var(--accent-gold);"></div>
                    <div class="waveform-bar" style="height: 38px; animation-delay: 0.2s;"></div>
                    <div class="waveform-bar" style="height: 18px; animation-delay: 0.5s;"></div>
                    <div class="waveform-bar" style="height: 42px; animation-delay: 0.4s; background: var(--accent-orange);"></div>
                    <div class="waveform-bar" style="height: 32px; animation-delay: 0.1s;"></div>
                    <div class="waveform-bar" style="height: 22px; animation-delay: 0.6s;"></div>
                    <div class="waveform-bar" style="height: 36px; animation-delay: 0.2s;"></div>
                    <div class="waveform-bar" style="height: 16px; animation-delay: 0.3s; background: var(--accent-gold);"></div>
                    <div class="waveform-bar" style="height: 30px; animation-delay: 0.5s;"></div>
                    <div class="waveform-bar" style="height: 40px; animation-delay: 0.1s;"></div>
                    <div class="waveform-bar" style="height: 24px; animation-delay: 0.4s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Dialogue Search Filter
            search_turn = st.text_input("🔍 Filter dialogue turns...", placeholder="Type to search phrases, keywords or speaker...", label_visibility="collapsed")

            # Dialogue Turns List
            filtered_turns = [
                t for t in turns 
                if not search_turn or search_turn.lower() in t.get('text', '').lower() or search_turn.lower() in t.get('speaker', '').lower()
            ]

            if filtered_turns:
                for idx, turn in enumerate(filtered_turns):
                    is_cust = "cust" in turn.get('speaker', '').lower() or turn.get('role') == 'customer'
                    speaker_class = "speaker-tag-customer" if is_cust else "speaker-tag-agent"
                    icon = "👤" if is_cust else "💼"
                    
                    highlight_html = f"<span class='highlight-chip'>{turn.get('highlight_tag')}</span>" if turn.get('is_highlight') and turn.get('highlight_tag') else ""
                    
                    st.markdown(f"""
                    <div class="turn-bubble">
                        <div class="speaker-row">
                            <span class="{speaker_class}">
                                <span>{icon}</span>
                                <span>{turn.get('speaker', 'Speaker')}</span>
                                {highlight_html}
                            </span>
                            <span class="turn-timestamp">{turn.get('timestamp', f'13:{40+idx}')}</span>
                        </div>
                        <div class="turn-text">{turn.get('text', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No conversational turns match your filter.")

            # Playback Toolbar (Matching Reference Image bottom bar)
            st.markdown("""
            <div class="player-bar">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="cursor: pointer; font-size: 1.2rem;">⏮</span>
                    <span style="cursor: pointer; font-size: 1.4rem; color: var(--primary-blue);">▶</span>
                    <span style="cursor: pointer; font-size: 1.2rem;">⏭</span>
                    <span style="font-size: 0.8rem; font-family: 'JetBrains Mono'; color: var(--text-muted); margin-left: 0.5rem;">03:42 / 13:56</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 0.78rem; font-weight: 700; background: var(--bg-warm); padding: 0.2rem 0.5rem; border-radius: 6px;">1.0x Speed</span>
                    <span style="cursor: pointer; font-size: 1.1rem;">🔊</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- RIGHT CANVAS: COACHING, OBJECTIONS & STRATEGY TABS ---
        with col_right_suite:
            st.markdown("### 💡 Intelligence & Coaching Suite")
            
            right_tab1, right_tab2, right_tab3, right_tab4 = st.tabs([
                "✨ Highlights", "🛡️ Objections", "📋 Action Items", "📊 Strategy"
            ])

            # Tab 1: Highlights & Key Topics
            with right_tab1:
                st.markdown("#### Primary Topic Tags")
                kw_html = " ".join([f"<span style='background: #EFF3FB; color: #3B5B99; padding: 0.25rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.82rem; margin-right: 0.4rem; display: inline-block; margin-bottom: 0.4rem;'>#{k}</span>" for k in res.get('keywords', [])])
                st.markdown(kw_html, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Notable Discussion Excerpts")
                highlighted_turns = [t for t in turns if t.get('is_highlight')]
                if highlighted_turns:
                    for ht in highlighted_turns:
                        st.markdown(f"""
                        <div style="background: #FFFFFF; border-left: 3px solid var(--accent-orange); padding: 0.6rem 0.85rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; font-size: 0.88rem;">
                            <strong>{ht.get('speaker')}:</strong> "{ht.get('text')}"
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No distinct highlight flags detected in standard turns.")

            # Tab 2: Objections & Battlecard Rebuttals
            with right_tab2:
                st.markdown("#### Customer Friction Points")
                objections = res.get('objections', [])
                if objections:
                    for obj in objections:
                        st.markdown(f"""
                        <div class="objection-card">
                            <div class="objection-title">⚠️ {obj.get('objection', 'Objection')} <span style="font-size: 0.75rem; color: #7E8B9B;">({obj.get('category', 'Sales')})</span></div>
                            <div class="rebuttal-box"><strong>Battlecard Strategy:</strong> {obj.get('recommended_rebuttal', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✓ No critical friction points or deal-breaking objections detected.")

            # Tab 3: Action Items Checklist
            with right_tab3:
                st.markdown("#### Post-Call Commitments")
                actions = res.get('action_items', [])
                if actions:
                    for act in actions:
                        p_color = "#EF4444" if act.get('priority') == 'High' else ("#F59E0B" if act.get('priority') == 'Medium' else "#10B981")
                        st.markdown(f"""
                        <div class="task-row">
                            <input type="checkbox" style="margin-top: 0.25rem;">
                            <div>
                                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.2rem;">
                                    <span class="task-assignee">{act.get('assignee', 'Rep')}</span>
                                    <span style="font-size: 0.72rem; font-weight: 700; color: {p_color};">● {act.get('priority', 'Normal')} Priority</span>
                                </div>
                                <div class="task-desc">{act.get('task', '')}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific action item checklist extracted.")

            # Tab 4: Executive Strategy
            with right_tab4:
                st.markdown(f"""
                <div class="dash-card">
                    {res.get('insights', '').replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# VIEW 2: CALL RECORDS ARCHIVE (SEARCHABLE CRM HISTORY)
# =============================================================================
elif nav_selection == "📁 Call Records Archive":
    st.markdown("## 📁 Call Records & CRM Archive")
    st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem;'>Browse historical calls, deal health classifications, and review past dialogues.</p>", unsafe_allow_html=True)
    
    calls = get_all_calls()
    
    if not calls:
        st.info("No recorded calls in database yet. Analyze your first call in the Live Studio!")
    else:
        df = pd.DataFrame([dict(c) for c in calls])
        
        # Search & Filter
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search_query = st.text_input("🔍 Search CRM records...", placeholder="Search intent, customer name or notes...")
        with col_f2:
            sentiment_filter = st.selectbox("Filter Sentiment", ["All", "Positive", "Neutral", "Negative"])
            
        filtered_df = df
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter]
        if search_query:
            filtered_df = filtered_df[filtered_df['transcript'].str.contains(search_query, case=False, na=False) | filtered_df['intent'].str.contains(search_query, case=False, na=False)]

        st.markdown(f"**Showing {len(filtered_df)} calls**")
        
        for idx, row in filtered_df.iterrows():
            with st.expander(f"📞 {row.get('intent', 'Call Interaction')} — {row.get('timestamp', '')[:16]} ({row.get('risk_level', 'Healthy')})"):
                st.markdown(f"**Sentiment:** `{row.get('sentiment_label')}` (Score: {row.get('sentiment_score', 0):.2f})")
                st.markdown(f"**Transcript Snippet:**\n> {row.get('transcript', '')[:280]}...")
                if st.button(f"🔍 Load Call #{row.get('id')} in Live Studio", key=f"load_call_{row.get('id')}"):
                    st.session_state['active_transcript'] = row.get('transcript', '')
                    st.session_state['run_pipeline_trigger'] = True
                    st.rerun()

# =============================================================================
# VIEW 3: COACHING & TEAM ANALYTICS
# =============================================================================
elif nav_selection == "📊 Coaching & Analytics":
    st.markdown("## 📊 Conversation Analytics & Team Coaching")
    st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem;'>High-level performance patterns, intent distribution, and sentiment trends across all calls.</p>", unsafe_allow_html=True)
    
    calls = get_all_calls()
    if not calls:
        st.info("Analyze multiple calls to populate aggregate coaching analytics.")
    else:
        df = pd.DataFrame([dict(c) for c in calls])
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # Intent Distribution Pie Chart
            fig_intent = px.pie(
                df,
                names='intent',
                title='Customer Intent & Call Categorization',
                hole=0.55,
                color_discrete_sequence=["#4667A7", "#F5BA72", "#E89A4B", "#5C7EBC", "#8EA7D1"]
            )
            fig_intent.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'family': 'Plus Jakarta Sans', 'color': '#323232'},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_intent, use_container_width=True)

        with col_c2:
            # Sentiment Score Trend
            df['date'] = pd.to_datetime(df['timestamp'])
            fig_trend = px.line(
                df.sort_values('date'),
                x='date',
                y='sentiment_score',
                title='Customer Sentiment Trajectory',
                markers=True,
                color_discrete_sequence=["#4667A7"]
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'family': 'Plus Jakarta Sans', 'color': '#323232'},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor="#EFE6DA")
            st.plotly_chart(fig_trend, use_container_width=True)

# =============================================================================
# VIEW 4: REFERENCE LIBRARY & DATASET EXPLORER
# =============================================================================
elif nav_selection == "📚 Reference Library":
    st.markdown("## 📚 Reference Dataset & Sample Dialogues")
    st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem;'>Browse DialogSum enterprise conversation samples and launch 1-click audits.</p>", unsafe_allow_html=True)
    
    samples = get_sample_conversations(4)
    for idx, sample in enumerate(samples):
        with st.container():
            st.markdown(f"""
            <div class="dash-card">
                <div style="font-weight: 700; font-size: 1rem; color: var(--primary-blue); margin-bottom: 0.5rem;">Sample Dialogue #{idx+1}</div>
                <div style="font-size: 0.9rem; color: #4B5563; line-height: 1.6; margin-bottom: 1rem;">
                    {sample['dialogue'].replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"🚀 Analyze Sample #{idx+1} in Live Studio", key=f"btn_sample_{idx}"):
                st.session_state['active_transcript'] = sample['dialogue']
                st.session_state['run_pipeline_trigger'] = True
                st.rerun()
