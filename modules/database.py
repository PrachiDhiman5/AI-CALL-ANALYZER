import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "call_analyzer.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create calls table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT,
            filename TEXT,
            transcript TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            intent TEXT,
            risk_level TEXT,
            keywords TEXT,
            insights TEXT,
            latency_metrics TEXT
        )
    ''')
    
    # Migration: Add risk_level column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE calls ADD COLUMN risk_level TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    conn.commit()
    conn.close()

def save_call_analysis(analysis_data):
    """
    Saves the analysis results to the database.
    analysis_data should be a dictionary.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO calls (
            source_type, filename, transcript, sentiment_score, 
            sentiment_label, intent, risk_level, keywords, insights, latency_metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        analysis_data.get('source_type'),
        analysis_data.get('filename'),
        analysis_data.get('transcript'),
        analysis_data.get('sentiment_score'),
        analysis_data.get('sentiment_label'),
        analysis_data.get('intent'),
        analysis_data.get('risk_level', "🟢 Healthy"),
        json.dumps(analysis_data.get('keywords', [])),
        analysis_data.get('insights'),
        json.dumps(analysis_data.get('latency_metrics', {}))
    ))
    
    conn.commit()
    conn.close()

def get_all_calls():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM calls ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
