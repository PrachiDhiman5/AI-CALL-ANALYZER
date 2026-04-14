from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

def extract_keywords(text, top_n=10):
    """
    Extracts key business terms from the transcript.
    Uses TF-IDF logic simplified for a single document context.
    """
    if not text:
        return []

    # Clean text: lowercase and remove non-alphanumeric
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # We use a custom set of common filler words to ignore
    stop_words = [
        'is', 'the', 'and', 'to', 'of', 'in', 'it', 'you', 'that', 'for', 
        'on', 'was', 'with', 'as', 'at', 'be', 'this', 'have', 'from', 'speaker',
        'person', 'hello', 'hi', 'um', 'uh', 'well', 'know', 'like', 'just'
    ]
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=50)
    
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Sort by score
        keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [word for word, score in keyword_scores[:top_n]]
    except:
        # Fallback to simple word frequency if TF-IDF fails (e.g., too short)
        words = [w for w in clean_text.split() if w not in stop_words and len(w) > 3]
        word_freq = pd.Series(words).value_counts()
        return word_freq.index[:top_n].tolist()

if __name__ == "__main__":
    sample = "The customer wants a refund for the broken laptop. The laptop was delayed in shipping."
    print(f"Keywords: {extract_keywords(sample)}")
