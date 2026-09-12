from sentence_transformers import SentenceTransformer
import time

class Embedder:
    def __init__(self):
        print("Initializing Embedding model (MiniLM)...")
        # Lightweight and extremely fast for production CPU usage
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(self, texts):
        """
        Converts a list of strings into a list of vectors.
        """
        start_time = time.time()
        embeddings = self.model.encode(texts)
        latency = (time.time() - start_time) * 1000
        return embeddings, round(latency, 2)

if __name__ == "__main__":
    emb = Embedder()
    vecs, latency = emb.get_embeddings(["Sample text for embedding"])
    print(f"Vector size: {len(vecs[0])}")
    print(f"Latency: {latency}ms")
