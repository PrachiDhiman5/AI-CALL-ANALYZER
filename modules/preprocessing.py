import re

def clean_transcript(text):
    """
    Cleans the transcript text for better NLP analysis.
    - Removes extra whitespace
    - Normalizes speaker tags (e.g., #Person1#: -> [Speaker 1]:)
    - Removes common filler words if necessary (optional for production)
    """
    if not text:
        return ""
    
    # 1. Normalize speaker tags found in DialogSum (#Person1#: -> [Speaker 1]:)
    text = re.sub(r'#Person1#:', '[Speaker A]:', text)
    text = re.sub(r'#Person2#:', '[Speaker B]:', text)
    text = re.sub(r'#Person3#:', '[Speaker C]:', text)
    
    # 2. Remove extra newlines and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Basic cleaning (stripping quotes, etc.)
    text = text.replace('"', '').replace("'", "")
    
    return text

def segment_by_speaker(text):
    """
    Splits the text into segments based on speakers.
    """
    # This is a bit more complex, for now we just return the full cleaned text
    return text

if __name__ == "__main__":
    raw = "#Person1#: Hello, how are you? \n\n #Person2#: I am fine, thank you!"
    print(f"Raw: {raw}")
    print(f"Cleaned: {clean_transcript(raw)}")
