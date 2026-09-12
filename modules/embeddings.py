import time
import numpy as np

class Embedder:
    def __init__(self):
        self.model = None
        self.dimension = 384
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Notice: SentenceTransformer fallback mode ({e}).")
            self.model = None

    def get_embeddings(self, texts):
        """
        Converts a list of strings into a list of vectors.
        """
        start_time = time.time()
        
        if self.model:
            try:
                embeddings = self.model.encode(texts)
                latency = (time.time() - start_time) * 1000
                return embeddings, round(latency, 2)
            except Exception:
                pass
                
        # Fast deterministic fallback vector generation
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for word in text.split():
                idx = abs(hash(word)) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
            
        latency = (time.time() - start_time) * 1000
        return np.array(embeddings, dtype=np.float32), round(latency, 2)

if __name__ == "__main__":
    emb = Embedder()
    vecs, latency = emb.get_embeddings(["Sample text for embedding"])
    print(f"Vector size: {len(vecs[0])}")
    print(f"Latency: {latency}ms")
