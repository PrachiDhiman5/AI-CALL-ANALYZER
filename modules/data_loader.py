from datasets import load_dataset
import pandas as pd
import os

def load_hf_dataset():
    """
    Loads the knkarthick/dialogsum dataset from Hugging Face.
    Returns a pandas DataFrame of the training set for demonstration.
    """
    print("Fetching dataset from Hugging Face...")
    try:
        # Load only the train split for now to keep it manageable
        dataset = load_dataset("knkarthick/dialogsum", split="train")
        df = pd.DataFrame(dataset)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def get_sample_conversations(n=10):
    """
    Returns n sample conversations from the dataset.
    """
    df = load_hf_dataset()
    if not df.empty:
        return df.sample(n).to_dict('records')
    return []

if __name__ == "__main__":
    samples = get_sample_conversations(2)
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        print(sample['dialogue'])
