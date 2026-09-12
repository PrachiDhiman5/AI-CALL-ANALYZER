import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import tempfile

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

# Page Config
st.set_page_config(page_title="AI Call Analyzer Pro", page_icon="🚀", layout="wide")

# Load environment variables
load_dotenv()
env_api_key = os.getenv("GROQ_API_KEY", "")

# Custom CSS: Zinc Executive Enterprise Theme
st.markdown("""
    <style>
    @import url('https://rsms.me/inter/inter.css');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #09090b !important;
        color: #fafafa !important;
    }
    
    /* Remove standard Streamlit padding */
    .block-container { padding-top: 2rem !important; max-width: 1200px !important; }
    
    /* Dash-Metric Cards (High Precision) */
    [data-testid="stMetric"] {
        background-color: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        padding: 1.2rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
    }
    
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; text-transform: uppercase; color: #a1a1aa !important; letter-spacing: 0.05em; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #ffffff !important; font-weight: 700 !important; }
    
    /* Section Headers */
    h1, h2, h3 { 
        font-weight: 700 !important; 
        color: #ffffff !important; 
        letter-spacing: -0.025em !important;
        border-bottom: none !important;
    }
    
    /* Sidebar Tweak: Professional Flat Zinc */
    [data-testid="stSidebar"] {
        background-color: #09090b !important;
        border-right: 1px solid #27272a !important;
    }
    
    /* Clean Buttons */
    .stButton>button {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.9 !important; }
    
    /* Card Styles with subtle hover */
    .insider-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        transition: border-color 0.3s, transform 0.2s;
    }
    .insider-card:hover {
        border-color: #3f3f46;
        transform: translateY(-2px);
    }
    
    /* Metrics Highlighting */
    [data-testid="stMetricValue"] { color: #10b981 !important; }
    
    /* Tab Styling: Minimalist Underline */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent !important; border-bottom: 1px solid #27272a !important; gap: 2rem !important; }
    .stTabs [data-baseweb="tab"] { color: #52525b !important; background-color: transparent !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom-color: #ffffff !important; }
    
    /* HIGH-CONTRASTY POLISH: Override Streamlit's Default Grey Labels */
    label[data-testid="stWidgetLabel"], .stSubheader, label, .stMarkdown p {
        color: #a5b4fc !important; /* Soft Indigo Blue */
        font-weight: 500 !important;
    }
    
    /* Ensure Insights text is WHITE for maximum readability on dark cards */
    .insider-card, .insider-card b, .insider-card div {
        color: #ffffff !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
    }
    
    /* Sidebar Arrow/Toggle Visibility */
    button[aria-label="Collapse sidebar"], button[aria-label="Expand sidebar"] {
        color: #10b981 !important; /* Emerald Green */
    }
    
    /* File Uploader Info Visibility */
    .stFileUploader section div div { color: #a5b4fc !important; }
    </style>
""", unsafe_allow_html=True)

# Initialization
@st.cache_resource
def load_models(version="2.1"): # Version bump forces a cache refresh
    # These models are downloaded on the first run (~500MB total)
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

# Better Visibility: Show what the app is doing during initialization
with st.spinner("Booting V2.1 Intelligence... (Loading models into memory)"):
    models = load_models(version="2.1")

# Sidebar: Stealth Security Layer
st.sidebar.title(" System Architecture")
st.sidebar.markdown("---")
st.sidebar.success(" **Status:** Stealth Mode Active")
st.sidebar.markdown("**Engine:** `Llama-3.3-70B-Versatile`")
st.sidebar.markdown("**Pipeline:** `Neural RAG v2.2` (Hybrid)")
st.sidebar.markdown("---")
if not env_api_key:
    st.sidebar.error("⚠️ `GROQ_API_KEY` missing from .env!")

# Reusable UI Components
def show_sentiment_gauge(score, label):
    import plotly.graph_objects as go
    
    colors = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#f43f5e"}
    color = colors.get(label, "#3f3f46")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Sentiment Score: {label}", 'font': {'size': 18, 'color': '#a5b4fc'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#71717a"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#27272a",
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        font={'color': "#fafafa", 'family': "Inter"}, 
        height=300, 
        margin=dict(l=30, r=30, t=50, b=20)
    )
    return fig

# Historical Context Toggle
if st.sidebar.button("Index Reference Dataset (DialogSum)"):
    with st.spinner("Indexing 100 sample calls for RAG..."):
        df = load_hf_dataset()
        if not df.empty:
            samples = df.head(100)
            texts = samples['dialogue'].tolist()
            embeddings, _ = models['embedder'].get_embeddings(texts)
            models['vector_db'].add_documents(embeddings, texts)
            st.sidebar.success(f"Indexed {len(texts)} conversations!")

# Main App Logic
st.title("AI Call Analyzer ")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Analyze Call", " Dataset Explorer", " Analytics & History"])

with tab1:
    st.header("Intelligence Dashboard")
    
    # Auto-analysis logic from Explorer
    if 'auto_analyze' in st.session_state and st.session_state['auto_analyze']:
        transcript_input = st.session_state['auto_analyze']
        st.session_state['auto_analyze'] = None # Clear it
        trigger_analysis = True
    else:
        transcript_input = ""
        trigger_analysis = False

    st.subheader("Data Ingestion")
    col_in1, col_in2 = st.columns([1, 1])
    
    with col_in1:
        input_type = st.radio("Mode", ["Text Entry", "Audio Upload"], horizontal=True)
        
        filename = "Internal_Entry"
        source_type = "Text"
        
        if input_type == "Audio Upload":
            audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"], label_visibility="collapsed")
            if audio_file:
                st.audio(audio_file)
                if st.button("Transcribe & Analyze"):
                    if not env_api_key: st.error("API Key missing!")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                            tmp.write(audio_file.getvalue())
                            tmp_path = tmp.name
                        with st.spinner("Processing..."):
                            transcript_input, _ = SpeechToText(env_api_key).transcribe(tmp_path)
                        os.remove(tmp_path)
                        source_type, filename = "Audio", audio_file.name
                        trigger_analysis = True
        else:
            transcript_input = st.text_area("Transcript Input", value=transcript_input, height=180, placeholder="Paste transcript...", label_visibility="collapsed")
            if st.button("Generate Report"):
                trigger_analysis = True

    if trigger_analysis and transcript_input:
        with st.status(" AI Pipeline Thinking...", expanded=True) as status:
            st.write("1.  Cleaning & Normalizing...")
            cleaned_text = clean_transcript(transcript_input)
            
            st.write("2.  Retrieving Context (RAG)...")
            # Setup Engines
            llm = LLMEngine(env_api_key)
            insights_engine = InsightsEngine(
                models['sentiment'], models['intent'], extract_keywords,
                models['embedder'], models['vector_db'], llm
            )
            
            st.write("3.  Running High-Density Models...")
            results = insights_engine.run_full_analysis(cleaned_text, filename, source_type)
            
            st.write("4.  Archiving Result...")
            save_call_analysis(results)
            status.update(label=" Analysis Complete", state="complete", expanded=False)
            
        # Display Results: Executive Dash
        st.markdown("---")
        st.header("🏢 Executive Intelligence Report")
        
        col_res1, col_res2 = st.columns([1, 1.2])
        
        with col_res1:
            st.plotly_chart(show_sentiment_gauge(results['sentiment_score'], results['sentiment_label']), use_container_width=True)
            
        with col_res2:
            st.subheader("Key Performance Indicators")
            m1, m2 = st.columns(2)
            m1.metric("Specific Intent", results['intent'])
            m2.metric("Sales Health", results['risk_level'])
            
            st.markdown("#### Primary Indicators")
            st.write(", ".join([f"`{k}`" for k in results['keywords']]))

        # Tabbed Insights Report
        st.markdown("---")
        res_tab1, res_tab2 = st.tabs(["💡 AI Strategy & Insights", "📄 Processed Data"])
        
        with res_tab1:
            st.markdown(f"""
                <div class="insider-card">
                    <b>Strategic Recommendations:</b><br><br>
                    {results['insights'].replace("\n", "<br>")}
                </div>
            """, unsafe_allow_html=True)
            
        with res_tab2:
            st.subheader("Telemetry & Metadata")
            st.json(results['latency_metrics'])
            st.subheader("Normalized Transcript")
            st.text_area("Final Cleaned Text", cleaned_text, height=200, disabled=True)

with tab2:
    st.header("Hugging Face Dataset Explorer")
    st.write("Browse genuine DialogSum conversations and analyze them instantly.")
    
    if st.button(" Fetch Random Samples"):
        samples = get_sample_conversations(5)
        st.session_state['samples'] = samples
        
    if 'samples' in st.session_state:
        for i, sample in enumerate(st.session_state['samples']):
            with st.expander(f" Sample Interaction {i+1}"):
                st.text_area(f"Transcript {i+1}", sample['dialogue'], height=150, disabled=True)
                if st.button(f" Analyze Sample {i+1}", key=f"btn_{i}"):
                    st.session_state['auto_analyze'] = sample['dialogue']
                    st.rerun() # Use rerun to switch tabs immediately

with tab3:
    st.header("Call Analytics & Persistent History")
    calls = get_all_calls()
    
    if not calls:
        st.info("No calls analyzed yet. History will appear here.")
    else:
        df_calls = pd.DataFrame([dict(c) for c in calls])
        
        # Analytics Visuals (Zinc Pro Theme)
        col_a, col_b = st.columns(2)
        
        zinc_palette = ["#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5"] # Emerald shades
        
        with col_a:
            # Intent Distribution
            fig_intent = px.pie(
                df_calls, 
                names='intent', 
                title='Primary Intent Distribution', 
                hole=0.6,
                color_discrete_sequence=zinc_palette
            )
            fig_intent.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#fafafa", 'family': "Inter"},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_intent, use_container_width=True)
            
        with col_b:
            # Sentiment Trend
            df_calls['timestamp'] = pd.to_datetime(df_calls['timestamp'])
            fig_sent = px.line(
                df_calls.sort_values('timestamp'), 
                x='timestamp', 
                y='sentiment_score', 
                title='Neural Sentiment Trend', 
                markers=True,
                color_discrete_sequence=["#10b981"]
            )
            fig_sent.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#fafafa", 'family': "Inter"},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig_sent.update_xaxes(showgrid=False, zeroline=False, color="#71717a")
            fig_sent.update_yaxes(showgrid=True, gridcolor="#27272a", zeroline=False, color="#71717a")
            st.plotly_chart(fig_sent, use_container_width=True)
        
        st.markdown("### 📜 Past Conversations")
        st.dataframe(df_calls[['timestamp', 'source_type', 'intent', 'risk_level', 'sentiment_label', 'sentiment_score']])
